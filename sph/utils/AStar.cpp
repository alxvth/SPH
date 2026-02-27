#include "AStar.hpp"

#include <sph/utils/Data.hpp>
#include <sph/utils/Distances.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/Logger.hpp>

#include <cassert>
#include <queue>
#include <utility>

namespace sph::utils {

    using KnnElement    = std::pair<float, int64_t>;
    using PriorityQueue = std::priority_queue<KnnElement, std::vector<KnnElement>, std::greater<KnnElement>> ;

    void astar(const GraphBaseInterface& graphView, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<int64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist, int verbosity)
    {
        if (verbosity >= 0)
            Log::info("Compute shortest path with A*");

        // Return -1 geodesic distance if no path was found
        geodesic.clear();
        g_dist = -1.f;

        PriorityQueue queue;
        queue.emplace(0.f, startID);

        // Define a distance map to store the shortest distance from the start vertex to each 
        distance_map.clear();
        distance_map.resize(data.getNumPoints(), -1.f);

        // Define a predecessor map to store the previous vertex from each vertex
        predecessor_map.clear();
        predecessor_map.resize(data.getNumPoints(), -1);

        predecessor_map[startID]    = startID;
        distance_map[startID]       = 0;

        int64_t current     = startID;
        float new_cost      = 0;
        int64_t new_neigh   = -1;
        float neigh_dist    = 0;

        while (!queue.empty()) {
            current = queue.top().second;

            if (current == endID) {
                break;
            }

            queue.pop();

            new_cost    = 0;
            new_neigh   = -1;
            neigh_dist  = 0;

            // start at 1 since 0 is the current point itself
            for (int64_t neighID = 1; neighID < graphView.getK(); neighID++) {

                neigh_dist  = graphView.getDistanceN(current, neighID);
                new_cost    = distance_map[current] + neigh_dist;

                if (neigh_dist <= 0)
                    continue;

                new_neigh = graphView.getNeighborN(current, neighID);

                if (distance_map[new_neigh] == -1.f || new_cost < distance_map[new_neigh])
                {
                    predecessor_map[new_neigh]  = current;
                    distance_map[new_neigh]     = new_cost;
                    
                    queue.emplace(new_cost + astarDistanceHeuristic(data, new_neigh, endID), new_neigh);
                }

            } // end for
        } // end while

        // geodesic path
        for (uint64_t v = endID; /* */; v = predecessor_map[v])
        {
            geodesic.emplace(geodesic.begin(), v);
            if (predecessor_map[v] == -1 || predecessor_map[v] == static_cast<int64_t>(v))
                break;
        }

        // geodesic distance
        g_dist = distance_map[endID];

        // print some info
        if (verbosity >= 1)
        {
            if (g_dist == -1.f)
                Log::warn("Geodesic could not be computed - probably the graph is not connected");

            Log::info("Geodesic between nodes: ({0}, {1})", startID, endID);
            Log::info("Euclid. dist : {}", astarDistanceHeuristic(data, startID, endID));
            Log::info("Geodesic dist: {}", g_dist);
        }

        if (verbosity >= 2)
        {
            Log::info("Accumulated distances on path:");
            for (uint64_t i = 0; i < geodesic.size() - 1; i++)
                Log::info("{:<8} -> {:<8}: {:.5g} (+ {:.5g})", geodesic[i], geodesic[i+1], distance_map[geodesic[i + 1]], astarDistanceHeuristic(data, geodesic[i], geodesic[i + 1]));
        }

#if SPH_DEBUG
        // geodesic distance cannot be shorter then direct L2 distance, up to some numerical precision
        float h = astarDistanceHeuristic(data, startID, endID);
        constexpr float eps = 0.0001f;
        if(g_dist != -1.f)
            assert(g_dist + eps >= h );
#endif // !SPH_DEBUG

    }

    // The heuristic here should also be admissible,
    // i.e. it never overestimates the actual cost to get to the endID
    // Here we use L2
    float astarDistanceHeuristic(const DataView& data, int64_t startID, int64_t endID)
    {
        assert(static_cast<size_t>(startID) * data.getNumDimensions() < data.getData().size());
        assert(static_cast<size_t>(endID)   * data.getNumDimensions() < data.getData().size());

        return utils::L2(data.data() + startID * data.getNumDimensions(), data.data() + endID * data.getNumDimensions(), data.getNumDimensions());

        //#include <faiss/utils/distances.hpp>
        //return utils::sqrtf(faiss::fvec_L2sqr(data.data() + startID * data.getNumDimensions(), data.data() + endID * data.getNumDimensions(), data.getNumDimensions()));
    }
}