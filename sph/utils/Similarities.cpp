#include "Similarities.hpp"

#include "Algorithms.hpp"
#include "CommonDefinitions.hpp"
#include "Data.hpp"
#include "DistanceCache.hpp"
#include "Distances.hpp"
#include "Graph.hpp"
#include "Hierarchy.hpp"
#include "Logger.hpp"
#include "Math.hpp" // invlin
#include "ShortestPath.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>

#include <Eigen/Dense>          // Eigen::MatrixXf
#include <Eigen/SparseCore>

#include <range/v3/view/enumerate.hpp>

namespace sph::utils {

    /// /////// ///
    /// Caching ///
    /// /////// ///
    namespace cache {
        utils::DistanceCache<int64_t, 4, float> cache = {};

        bool isActive() { return cache.getUseCacheDistanceCache(); }
        bool isSymmetricLookup() { return cache.getUseSymmetricLookupDistanceCache(); }

        void setActive(bool c) {
            cache.setUseCacheDistanceCache(c);
            Log::info("utils::cache::setActive: set to {}", isActive());
        }

        void setSymmetricLookup(bool c) {
            cache.setUseSymmetricLookupDistanceCache(c);
            Log::info("utils::cache::setSymmetricLookup: set to {}", isSymmetricLookup());
        }

        void clear() {
            Log::info("utils::cache::clear: size {}", cache._cacheDistance.size());
            cache.clearCacheDistanceCache();
        }

        void reserve(size_t capacity) {
            cache.reserve(capacity);
        }

        static void add(const ComponentID& cid1, const ComponentID& cid2, float dist) {
            cache.add(std::make_tuple(cid1.level, cid1.id, cid2.level, cid2.id), dist);
        }

        static bool contains(const ComponentID& cid1, const ComponentID& cid2, float& dist) {
            return cache.contains(std::make_tuple(cid1.level, cid1.id, cid2.level, cid2.id), dist);
        }
    }

    /// ////////////////////// ///
    /// SimilaritiesStatistics ///
    /// ////////////////////// ///

    namespace stats {

        std::atomic<int64_t> _stats_NumCallTotal = 0;
        std::atomic<int64_t> _stats_NumComputeTotal = 0;
        std::atomic<int64_t> _stats_NumCacheLookupSuccess = 0;

        SimilaritiesStatistics getSimilaritiesStatistics() { return { _stats_NumCallTotal, _stats_NumComputeTotal, _stats_NumCacheLookupSuccess }; }

        inline static void IncrCallTotal() { ++_stats_NumCallTotal; }
        inline static void IncrComputeTotal() { ++_stats_NumComputeTotal; }
        inline static void IncrCacheLookupSuccess() { ++_stats_NumCacheLookupSuccess; }
    }

    /// //////////// ///
    /// Similarities ///
    /// //////////// ///

    [[nodiscard]] float componentSimilarity(const ComponentSim& cs, const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, const std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents)
    {
        stats::IncrCallTotal();

        if (cid1 == cid2)
            return 1.f;

        float sim = 0.f;
        
        if (cache::isActive() && cache::contains(cid1, cid2, sim))
        {
            stats::IncrCacheLookupSuccess();
            return sim;
        }

        stats::IncrComputeTotal();

        switch (cs)
        {
        case ComponentSim::GEO_CENTROID:                sim = simGeodesicDistance(hierarchy, knnDataLevel, data, cid1, cid2, connectedComponents);  break;
        case ComponentSim::NEIGH_OVERLAP:               sim = simNeighborOverlap(hierarchy, knnDataLevel, cid1, cid2);                              break;
        case ComponentSim::NEIGH_WALKS:                 sim = simRandomWalksBhattacharyya(hierarchy, cid1, cid2);                                   break;
        case ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP:  sim = simRandomWalksSingleOverlay(hierarchy, cid1, cid2);                                   break;
        case ComponentSim::GEO_WALKS:                   sim = simGeodesicDistance(hierarchy, knnDataLevel, data, cid1, cid2, connectedComponents);  break;
        case ComponentSim::EUCLID_CENTROID:             sim = simEuclidDistance(hierarchy, data, cid1, cid2);                                       break;
        }

        assert(sim >= 0);
        assert(sim <= 1);

        if (cache::isActive())
            cache::add(cid1, cid2, sim);

        return sim;
    }

    [[nodiscard]] float componentDistance(const ComponentSim& cs, const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, const std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents)
    {
        stats::IncrCallTotal();

        if (cid1 == cid2)
            return 0.f;

        float dist = 0.f;
        
        if (cache::isActive() && cache::contains(cid1, cid2, dist))
        {
            stats::IncrCacheLookupSuccess();
            return dist;
        }

        stats::IncrComputeTotal();

        switch (cs)
        {
        case ComponentSim::GEO_CENTROID:                dist = geodesicDistance(hierarchy, knnDataLevel, data, cid1, cid2, connectedComponents);  break;
        case ComponentSim::NEIGH_OVERLAP:               dist = 1.f - simNeighborOverlap(hierarchy, knnDataLevel, cid1, cid2);                     break;
        case ComponentSim::NEIGH_WALKS:                 dist = 1.f - simRandomWalksBhattacharyya(hierarchy, cid1, cid2);                          break;
        case ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP:  dist = 1.f - simRandomWalksSingleOverlay(hierarchy, cid1, cid2);                          break;
        case ComponentSim::GEO_WALKS:                   dist = geodesicDistance(hierarchy, knnDataLevel, data, cid1, cid2, connectedComponents);  break;
        case ComponentSim::EUCLID_CENTROID:             dist = euclidDistance(hierarchy, data, cid1, cid2);                                       break;
        }

        assert(dist >= 0);

        if (cache::isActive())
            cache::add(cid1, cid2, dist);

        return dist;
    }

    /// //////////////////// ///
    /// Neighborhood Overlap ///
    /// //////////////////// ///

    // Helper for counting overlap size without allocating overlap vector
    struct IntersectionCounter {
        IntersectionCounter() : s(0), count(0) {}
        IntersectionCounter& operator++()
        {
            ++count;
            return *this;
        }
        uint64_t& operator*() { return s; }
        uint64_t s, count;
    };

    [[nodiscard]] uint64_t representedOverlap(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const ComponentID& cid1, const ComponentID& cid2, std::vector<uint64_t>& unionNeighbors1, std::vector<uint64_t>& unionNeighbors2)
    {
        unionNeighbors1.clear();
        unionNeighbors2.clear();

        std::vector<uint64_t> repIDs1, repIDs2;

        // Represented data points of components 1 and 2
        hierarchy.getRepresentedDataPoints(cid1, repIDs1);
        hierarchy.getRepresentedDataPoints(cid2, repIDs2);

        if (repIDs1.size() == 0 || repIDs2.size() == 0)
        {
            Log::warn("representedOverlap:: No represented data level IDs for {0}-{1} or {2}-{3}", cid1.level, cid1.id, cid2.level, cid2.id);
            return 0;
        }

        // Get unique knn ids
        auto getKnn = [&knnDataLevel](const std::vector<uint64_t>& repIDs, std::vector<uint64_t>& unionNeighbors) {
            // Retrieve all knn
            for (const auto& repID : repIDs)
            {
                const auto KnnSpan = knnDataLevel->getNeighbors(repID);
                std::ranges::copy(KnnSpan.begin(), KnnSpan.end(), std::back_inserter(unionNeighbors));
            }
            
            utils::sortAndUnique(unionNeighbors);
            };

        // Nearest neighbors of components 1 and 2
        getKnn(repIDs1, unionNeighbors1);
        getKnn(repIDs2, unionNeighbors2);

        // Overlap of the neighborhood
        IntersectionCounter const& intersection = std::set_intersection(
            unionNeighbors1.begin(), unionNeighbors1.end(),
            unionNeighbors2.begin(), unionNeighbors2.end(),
            IntersectionCounter());

        return intersection.count;
    }   

    [[nodiscard]] float simNeighborOverlap(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const ComponentID& cid1, const ComponentID& cid2)
    {        
        assert(knnDataLevel->getKnnIndices().size() > 0);

        std::vector<uint64_t> unionNeighbors1, unionNeighbors2;
        uint64_t overlapSize =  representedOverlap(hierarchy, knnDataLevel, cid1, cid2, unionNeighbors1, unionNeighbors2);

        auto minSize = (unionNeighbors1.size() <= unionNeighbors2.size()) ? unionNeighbors1.size() : unionNeighbors2.size();

        float sim = (minSize > 0) ? static_cast<float>(overlapSize) / minSize : 0;

        return sim;
    }

    /// ///////////////// ///
    /// Geodesic distance ///
    /// ///////////////// ///

    [[nodiscard]] float geodesicDistance(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents)
    {
        // if level is == 0 just use the shortest path between them
        // otherwise get the represented data points
        // If it's more than hierarchy.settings.numGeodesicSamples**2, sample the two sets

        auto geoDist = [&knnDataLevel, &data, &connectedComponents, &hierarchy](int64_t startID, int64_t endID) -> float {
            return utils::computeShortestPath(*knnDataLevel, data, startID, endID, hierarchy.bgraph.get(), connectedComponents);
        };

        if (cid1.level == 0 && cid2.level == 0)
        {
            float dist = geoDist(cid1.id, cid2.id);

            if (dist == -1.f) [[unlikely]]
                return std::numeric_limits<float>::max();

            return dist;
        }

        // Represented data points of components 1 and 2
        std::vector<int64_t> repIDs1, repIDs2;
        hierarchy.getRepresentedDataPoints(cid1, repIDs1);
        hierarchy.getRepresentedDataPoints(cid2, repIDs2);

        const size_t numSamples     = hierarchy.settings.numGeodesicSamples;
        const size_t numComparisons = (numSamples == std::numeric_limits<size_t>::max()) ? std::numeric_limits<size_t>::max() : numSamples * numSamples;

        Eigen::MatrixXf distanceMatrix;

        auto computeDistances = [&distanceMatrix, &geoDist](std::vector<int64_t>& smpls1, std::vector<int64_t>& smpls2) -> void {

            for (const auto& [id1_pos, id1_val] : ranges::views::enumerate(smpls1))
            {
                for (const auto& [id2_pos, id2_val] : ranges::views::enumerate(smpls2))
                {
                    float dist = geoDist(id1_val, id2_val);
                    if (dist >= 0)
                        distanceMatrix(id1_pos, id2_pos) = dist;
                    else [[unlikely]]
                        distanceMatrix(id1_pos, id2_pos) = std::numeric_limits<float>::max();

                }
            }
        };

        // Compute all distance pairs if number of combinations is smaller sample size
        if (repIDs1.size() * repIDs2.size() <= numComparisons) [[likely]]
        {
            distanceMatrix = Eigen::MatrixXf::Zero(repIDs1.size(), repIDs2.size());
            computeDistances(repIDs1, repIDs2);
        }
        else [[unlikely]]
        {
            std::vector<int64_t> samples1, samples2;
            distanceMatrix = Eigen::MatrixXf::Zero(numSamples, numSamples);

            auto generateSamples = [numSamples](auto& vals, auto& samples) -> void {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<size_t> dist(0, vals.size() - 1);

                samples.reserve(numSamples);
                for (size_t i = 0; i < numSamples; ++i)
                    samples.push_back(vals[dist(gen)]);
            };

            generateSamples(repIDs1, samples1);
            generateSamples(repIDs2, samples2);

            computeDistances(samples1, samples2);
        }

        float dist = symmetricHausdorffDistance(distanceMatrix);

        return dist;
    }

    [[nodiscard]] float simGeodesicDistance(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents)
    {
        float sim = geodesicDistance(hierarchy, knnDataLevel, data, cid1, cid2, connectedComponents);

        if (sim == std::numeric_limits<float>::max())
            sim = 0;
        else
            sim = invlin(sim);

        return sim;
    }

    /// //////////// ///
    /// Random Walks ///
    /// //////////// ///

    [[nodiscard]] float simRandomWalksSingleOverlay(const Hierarchy& hierarchy, const ComponentID& cid1, const ComponentID& cid2)
    {
        float sim = 0.f;

        if (cid1.level != cid2.level)
        {
            Log::warn("simRandomWalks: cid1.level != cid2.level. Not implemented/possible. Returning 0.");
            return sim;

            // Implementation idea:
            // Look if the random walk ended on the parents, 
            // i.e. make use of the children/parents maps

        }

        assert(cid1.level == cid2.level);

        float sim12 = randomWalksSingleOverlap(hierarchy.randomWalks, cid1.level, cid1.id, cid2.id);
        float sim21 = randomWalksSingleOverlap(hierarchy.randomWalks, cid1.level, cid2.id, cid1.id);

        sim = std::max(sim12, sim21);

        return sim;
    }

    [[nodiscard]] float simRandomWalksBhattacharyya(const Hierarchy& hierarchy, const ComponentID& cid1, const ComponentID& cid2)
    {
        float sim = 0.f;

        if (cid1.level != cid2.level)
        {
            Log::warn("simRandomWalks: cid1.level != cid2.level. Not implemented/possible. Returning 0.");
            return sim;

            // Implementation idea:
            // Look if the random walk ended on the parents, 
            // i.e. make use of the children/parents maps
        }

        assert(cid1.level == cid2.level);

        sim = randomWalksBhattacharyya(hierarchy.randomWalks, cid1.level, cid1.id, cid2.id);

        return sim;
    }

    [[nodiscard]] float randomWalksSingleOverlap(const std::vector<std::vector<SparseVecSPH>>& similarities, std::uint64_t level, std::uint64_t id1, std::uint64_t id2)
    {
        return similarities[level][id1].coeff(id2);
    }

    [[nodiscard]] float randomWalksBhattacharyya(const std::vector<std::vector<SparseVecSPH>>& similarities, std::uint64_t level, std::uint64_t id1, std::uint64_t id2)
    {
        // assumes probDist1 and probDist2 are normalized
        const auto& probDist1 = similarities[level][id1];
        const auto& probDist2 = similarities[level][id2];

        float bc = 0.f;

        for (SparseVecSPH::InnerIterator it1(probDist1); it1; ++it1) {
            assert(it1.value() > 0.0f);
            const float value2 = probDist2.coeff(it1.index()); // get the corresponding value from normDist2, 0 if not present
            if (value2 > 0.0f) {
                bc += std::sqrt(it1.value() * value2);
            }
        }

        return bc;
    }

    /// /////////////// ///
    /// Euclid Distance ///
    /// /////////////// ///

    [[nodiscard]] float simEuclidDistance(const Hierarchy& hierarchy, const DataView& data, const ComponentID& cid1, const ComponentID& cid2)
    {
        float sim = euclidDistance(hierarchy, data, cid1, cid2);

        if (sim == std::numeric_limits<float>::max())
            sim = 0;
        else
            sim = invlin(sim);

        return sim;
    }

    [[nodiscard]] float euclidDistance(const Hierarchy& hierarchy, const DataView& data, const ComponentID& cid1, const ComponentID& cid2)
    {
        // if level is == 0 just use the shortest path between them
        // otherwise get the represented data points
        // If it's more than hierarchy.settings.numGeodesicSamples**2, sample the two sets

        auto euclidDist = [&data](int64_t startID, int64_t endID) -> float {
            auto x1 = data.getValuesAt(startID);
            auto x2 = data.getValuesAt(endID);
            return L2(x1.data(), x2.data(), data.getNumDimensions());
            };

        if (cid1.level == 0 && cid2.level == 0)
        {
            float dist = euclidDist(cid1.id, cid2.id);

            return dist;
        }

        // Represented data points of components 1 and 2
        std::vector<int64_t> repIDs1, repIDs2;
        hierarchy.getRepresentedDataPoints(cid1, repIDs1);
        hierarchy.getRepresentedDataPoints(cid2, repIDs2);

        const size_t numSamples = hierarchy.settings.numGeodesicSamples;
        const size_t numComparisons = (numSamples == std::numeric_limits<size_t>::max()) ? std::numeric_limits<size_t>::max() : numSamples * numSamples;

        Eigen::MatrixXf distanceMatrix;

        auto computeDistances = [&distanceMatrix, &euclidDist](std::vector<int64_t>& smpls1, std::vector<int64_t>& smpls2) -> void {
            for (const auto& [id1_pos, id1_val] : ranges::views::enumerate(smpls1))
                for (const auto& [id2_pos, id2_val] : ranges::views::enumerate(smpls2))
                    distanceMatrix(id1_pos, id2_pos) = euclidDist(id1_val, id2_val);
            };

        // Compute all distance pairs if number of combinations is smaller sample size
        if (repIDs1.size() * repIDs2.size() <= numComparisons) [[likely]]
        {
            distanceMatrix = Eigen::MatrixXf::Zero(repIDs1.size(), repIDs2.size());
            computeDistances(repIDs1, repIDs2);
        }
        else [[unlikely]]
        {
            std::vector<int64_t> samples1, samples2;
            distanceMatrix = Eigen::MatrixXf::Zero(numSamples, numSamples);

            auto generateSamples = [numSamples](auto& vals, auto& samples) -> void {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<size_t> dist(0, vals.size() - 1);

                samples.reserve(numSamples);
                for (size_t i = 0; i < numSamples; ++i)
                    samples.push_back(vals[dist(gen)]);
                };

            generateSamples(repIDs1, samples1);
            generateSamples(repIDs2, samples2);

            computeDistances(samples1, samples2);
        }

        // TODO: Think about the distance here again
        // Maybe also mean
        //std::cout << "Matrix:\n" << distanceMatrix << "\n\n";

        float dist = symmetricHausdorffDistance(distanceMatrix);

        return dist;
    }


} // namespace sph::utils
