#include "ShortestPath.hpp"

#include <atomic>
#include <ranges>
#include <tuple>

#include "AStar.hpp"
#include "AStarBoost.hpp"
#include "DistanceCache.hpp"
#include "Logger.hpp"

namespace sph::utils {

    /// ////////////////////// ///
    /// ShortestPathStatistics ///
    /// ////////////////////// ///

    namespace stats {

        std::atomic<int64_t> _spNumCallTotal = 0;
        std::atomic<int64_t> _spNumComputeTotal = 0;
        std::atomic<int64_t> _spNumComputeAStar = 0;
        std::atomic<int64_t> _spNumComputeBoostAStar = 0;
        std::atomic<int64_t> _spNumComputeBoostDijkstra = 0;
        std::atomic<int64_t> _spNumCacheLookupSuccess = 0;

        ShortestPathStatistics getShortestPathStatistics() { return { _spNumCallTotal, _spNumComputeTotal, _spNumComputeAStar, _spNumComputeBoostAStar, _spNumComputeBoostDijkstra, _spNumCacheLookupSuccess }; }

        inline static void _spCallTotal() { ++_spNumCallTotal; }
        inline static void _spComputeTotal() { ++_spNumComputeTotal; }
        inline static void _spComputeAStar() { ++_spNumComputeAStar; }
        inline static void _spComputeBoostAStar() { ++_spNumComputeBoostAStar; }
        inline static void _spComputeBoostDijkstra() { ++_spNumComputeBoostDijkstra; }
        inline static void _spCacheLookupSuccess() { ++_spNumCacheLookupSuccess; }
    }

    /// /////// ///
    /// Caching ///
    /// /////// ///
    namespace cache {
        using ShortestPathCache = utils::DistanceCache<int64_t, 2, float>;
        ShortestPathCache shortestPathCache = {};

        bool getUseCacheShortestPath() { return shortestPathCache.getUseCacheDistanceCache(); }
        bool getUseSymmetricLookupShortestPath() { return shortestPathCache.getUseSymmetricLookupDistanceCache(); }

        void setUseCacheShortestPath(bool c) {
            shortestPathCache.setUseCacheDistanceCache(c);
            Log::info("utils::cache::setUseCacheShortestPath: set to {}", getUseCacheShortestPath());
        }

        void setUseSymmetricLookupShortestPath(bool c) {
            shortestPathCache.setUseSymmetricLookupDistanceCache(c);
            Log::info("utils::cache::setUseSymmetricLookupShortestPath: set to {}", getUseSymmetricLookupShortestPath());
        }

        void clearCacheShortestPath() {
            Log::info("utils::cache::clearCacheShortestPath: size {}", shortestPathCache._cacheDistance.size());
            shortestPathCache.clearCacheDistanceCache();
        }

        void reserveCacheShortestPath(size_t capa) {
            shortestPathCache.reserve(capa);
        }

        static void add(int64_t startID, int64_t endID, float dist) {
            shortestPathCache.add(std::make_tuple(startID, endID), dist);
        }

        static bool contains(int64_t startID, int64_t endID, float& dist) {
            return shortestPathCache.contains(std::make_tuple(startID, endID), dist);
        }

        static void addGeodesic(const std::vector<int64_t>& geodesic, const std::vector<float>& distance_map, int64_t startID) {
            // cache distances from start to all points along and between points on the geodesic
            if (getUseCacheShortestPath())
            {
                const size_t skipOuter = 1;
                size_t       skipInner = 2;

                for (const auto& stepID : geodesic | std::views::drop(skipOuter))
                {
                    float stepDist = distance_map[stepID];
                    add(startID, stepID, stepDist);

                    for (const auto& InnerStepID : geodesic | std::views::drop(skipInner))
                        add(stepID, InnerStepID, distance_map[InnerStepID] - stepDist);

                    skipInner++;
                }

            }
        }
    }

    /// ///////////// ///
    /// Shortest path ///
    /// ///////////// ///

    float computeShortestPath(const GraphBaseInterface& knnDataLevel, const DataView& data, int64_t startID, int64_t endID, const utils::BoostGraph* boostGraph, std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents)
    {
        stats::_spCallTotal();
        float g_dist = -1.f;

        if (cache::getUseCacheShortestPath() && cache::contains(startID, endID, g_dist))
        {
            stats::_spCacheLookupSuccess();

            //appendLineToFile("shortestPath_pathSymCacheNoSym.txt", fmt::format("({0}, {1}): {2}", startID, endID, g_dist));
            return g_dist;
        }

        // if start and end point are in different connected components, there cannot be a connection
        if (connectedComponents.has_value() && connectedComponents.value().get() && connectedComponents.value()->size() == static_cast<size_t>(data.getNumPoints()))
        {
            auto& wwc = connectedComponents.value();
            if (wwc->at(startID) != wwc->at(endID))
            {
                if (cache::getUseCacheShortestPath())
                    cache::add(startID, endID , g_dist);

                return g_dist;
            }
        }

        // check if nodes are direct neighbors, also returns if startID == endID
        int64_t dir_neigh = knnDataLevel.isDirectNeighbor(startID, endID);
        if (dir_neigh >= 0)
        {
            g_dist = knnDataLevel.getDistanceN(startID, dir_neigh);

            //appendLineToFile("shortestPath_pathSymCacheNoSym.txt", fmt::format("({0}, {1}): {2}", startID, endID, g_dist));
            return g_dist;
        }

        std::vector<int64_t> geodesic_temp;
        std::vector<float> distance_map_temp;
        std::vector<uint64_t> predecessor_map_temp;

        stats::_spComputeTotal();

        if (boostGraph)
        {
            astarBoost(*boostGraph, data, startID, endID, distance_map_temp, predecessor_map_temp, geodesic_temp, g_dist);
            stats::_spComputeBoostAStar();

            if (g_dist == -1.f)
            {
                dijkstraBoost(*boostGraph, startID, endID, distance_map_temp, predecessor_map_temp, geodesic_temp, g_dist);
                stats::_spComputeBoostDijkstra();
            }

        }
        else
        {
            astar(knnDataLevel, data, startID, endID, distance_map_temp, geodesic_temp, g_dist);
            stats::_spComputeAStar();
        }

        // add values to cache
        cache::addGeodesic(geodesic_temp, distance_map_temp, startID);

        //appendLineToFile("shortestPath_pathSymCacheNoSym.txt", fmt::format("({0}, {1}): {2}", startID, endID, g_dist));
        return g_dist;

    }

}
