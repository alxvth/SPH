#pragma once

#include <cstdint>
#include <vector>

namespace sph::utils
{
    struct GraphBaseInterface;
    struct DataView;

    void astar(const GraphBaseInterface& graphView, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<int64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist, int verbosity = -1);

    inline void astar(const GraphBaseInterface& graphView, const DataView& data, int64_t startID, int64_t endID, std::vector<int64_t>& geodesic, float& g_dist, int verbosity = -1)
    {
        std::vector<float> distance_map;
        std::vector<int64_t> predecessor_map;

        astar(graphView, data, startID, endID, distance_map, predecessor_map, geodesic, g_dist, verbosity);
    }

    inline void astar(const GraphBaseInterface& graphView, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<int64_t>& geodesic, float& g_dist, int verbosity = -1)
    {
        std::vector<int64_t> predecessor_map;

        astar(graphView, data, startID, endID, distance_map, predecessor_map, geodesic, g_dist, verbosity);
    }

    float astarDistanceHeuristic(const DataView& data, int64_t someID, int64_t endID);
}
