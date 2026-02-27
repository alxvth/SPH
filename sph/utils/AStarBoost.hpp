#pragma once

#include <cstdint>
#include <vector>

#include "Graph.hpp"
#include "GraphBoost.hpp"

namespace sph::utils
{
    struct DataView;

    void astarBoost(const BoostGraph& bgraph, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist);
    void astarBoost(const GraphView& graphView, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist);

    void dijkstraBoost(const BoostGraph& bgraph, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist);

}
