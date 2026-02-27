#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "AStarBoost.hpp"
#include "Data.hpp"
#include "Graph.hpp"

namespace sph::utils
{
    // uses boost implementation of A* if hierarchy.bgraph is populated, tries Dijkstra if A* fails
    // otherwise falls back to own implementation of A*
    // returns -1 if no connection in the graph can be found
    float computeShortestPath(const GraphBaseInterface& knnDataLevel, const DataView& data, int64_t startID, int64_t endID, const utils::BoostGraph* boostGraph, std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents);

    namespace cache {
        void clearCacheShortestPath();
        void reserveCacheShortestPath(size_t capa);
        void setUseCacheShortestPath(bool c);
        void setUseSymmetricLookupShortestPath(bool c);
        bool getUseCacheShortestPath();
        bool getUseSymmetricLookupShortestPath();
    }

    namespace stats {
        struct ShortestPathStatistics {
            int64_t numCallTotal = 0;
            int64_t numComputeTotal = 0;
            int64_t numComputeAStar = 0;
            int64_t numComputeBoostAStar = 0;
            int64_t numComputeBoostDijkstra = 0;
            int64_t numCacheLookupSuccess = 0;
        };

        ShortestPathStatistics getShortestPathStatistics();
    }

}
