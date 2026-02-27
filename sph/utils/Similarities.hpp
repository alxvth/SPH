#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "Settings.hpp"

namespace Eigen
{
    template <typename Scalar, int Options, typename Index>
    class SparseVector;
}

namespace sph::utils {

    struct DataView;
    struct Hierarchy;
    struct GraphBaseInterface;
    struct ComponentID;

    // sim will be in [0, 1] with 1 meaning most similar (e.g. identical) and 0 meaning least similar (e.g. no overlap in neighborhoods or infinite geodesic distance)
    [[deprecated("Might be removed in the future - use componentDistance instead")]]
    [[nodiscard]] float componentSimilarity(const ComponentSim& cs, const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, const std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents = std::nullopt);

    // similar to componentSimilarity: does not invert the distance to similarities but inverts similarities to distances
    [[nodiscard]] float componentDistance(const ComponentSim& cs, const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, const std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents = std::nullopt);

    // Computes the similarity between two nodes wrt to the overlap of their neighbors. sim will be in [0,1]
    [[nodiscard]] float simNeighborOverlap(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const ComponentID& cid1, const ComponentID& cid2);

    // Helper for simNeighborOverlap
    [[nodiscard]] uint64_t representedOverlap(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const ComponentID& cid1, const ComponentID& cid2, std::vector<uint64_t>& unionNeighbors1, std::vector<uint64_t>& unionNeighbors2);

    // Computes the similarity between two nodes wrt to their geodesic distance (length of shortest path) on the knn Graph, inverts distance to sim with 1 / (1 + dist) such that sim will be in [0,1]
    [[nodiscard]] float simGeodesicDistance(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents = std::nullopt);

    // Computes the geodesic distance between two nodes (length of shortest path) on the knn Graph
    // For sets of points, computes the Hausdorff distance with geodesic distance as it's base
    [[nodiscard]] float geodesicDistance(const Hierarchy& hierarchy, const GraphBaseInterface* knnDataLevel, const DataView& data, const ComponentID& cid1, const ComponentID& cid2, std::optional<std::shared_ptr<std::vector<int64_t>>> connectedComponents = std::nullopt);

    // Computes the similarity 
    [[nodiscard]] float simEuclidDistance(const Hierarchy& hierarchy, const DataView& data, const ComponentID& cid1, const ComponentID& cid2);

    // Computes the 
    [[nodiscard]] float euclidDistance(const Hierarchy& hierarchy, const DataView& data, const ComponentID& cid1, const ComponentID& cid2);

    // Computes the similarity between two nodes wrt to their random walks on the knn Graph, assumes normalized random walks, such that sim will be in [0,1]
    // Checks if random walks form id1 and id2 overlap anywhere, uses that value as similarity
    [[nodiscard]] float simRandomWalksSingleOverlay(const Hierarchy& hierarchy, const ComponentID& cid1, const ComponentID& cid2);

    // Computes the similarity between two nodes wrt to their random walks on the knn Graph, assumes normalized random walks, such that sim will be in [0,1]
    // Computes Bhattacharyya distances of all overlapping random walk entries
    [[nodiscard]] float simRandomWalksBhattacharyya(const Hierarchy& hierarchy, const ComponentID& cid1, const ComponentID& cid2);
    
    // assumes similarities are normalized to proper probability distributions, distance helper for simRandomWalksSingleOverlay
    [[nodiscard]] float randomWalksSingleOverlap(const std::vector<std::vector<SparseVecSPH>>& similarities, std::uint64_t level, std::uint64_t id1, std::uint64_t id2);

    // assumes similarities are normalized to proper probability distributions, distance helper for simRandomWalksBhattacharyya
    [[nodiscard]] float randomWalksBhattacharyya(const std::vector<std::vector<SparseVecSPH>>& similarities, std::uint64_t level, std::uint64_t id1, std::uint64_t id2);

    namespace cache {
        void clear();
        void reserve(size_t capacity);
        void setActive(bool c);
        void setSymmetricLookup(bool c);
        bool isActive();
        bool isSymmetricLookup();
    }

    namespace stats {
        struct SimilaritiesStatistics {
            int64_t numCallTotal = 0;
            int64_t numComputeTotal = 0;
            int64_t numCacheLookupSuccess = 0;
        };

        SimilaritiesStatistics getSimilaritiesStatistics();
    }

} // namespace sph::utils


