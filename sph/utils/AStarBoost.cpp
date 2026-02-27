#include "AStarBoost.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <sph/utils/Data.hpp>
#include <sph/utils/Distances.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/GraphAdapterBoost.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/ProgressBar.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/detail/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>

namespace sph::utils
{
    // Custom heuristic: euclidean distance
    class euclidean_heuristic
        : public boost::astar_heuristic< BoostGraph, float >
    {
    public:
        euclidean_heuristic(BoostGraph::vertex_descriptor endID, const std::vector<float>& pointData, uint64_t numDimensions) : 
            _endID(endID), _pointData(pointData), _numDimensions(numDimensions) {};

        float operator()(BoostGraph::vertex_descriptor v)
        {
            return std::sqrt(utils::L2Sqr(_pointData.data() + v * _numDimensions, _pointData.data() + _endID * _numDimensions, _numDimensions));
        }

    private:
        BoostGraph::vertex_descriptor _endID;
        const std::vector<float>& _pointData;
        uint64_t _numDimensions;
    };

    class euclidean_heuristic_sph_graph
        : public boost::astar_heuristic< GraphView, float >
    {
    public:
        euclidean_heuristic_sph_graph(boost::graph_traits<GraphView>::vertex_descriptor endID, const std::vector<float>& pointData, uint64_t numDimensions) :
            _endID(endID), _pointData(pointData), _numDimensions(numDimensions) {};

        float operator()(boost::graph_traits<GraphView>::vertex_descriptor v)
        {
            return std::sqrt(utils::L2Sqr(_pointData.data() + v * _numDimensions, _pointData.data() + _endID * _numDimensions, _numDimensions));
        }

    private:
        boost::graph_traits<GraphView>::vertex_descriptor _endID;
        const std::vector<float>& _pointData;
        uint64_t _numDimensions;
    };


    // exception for termination
    struct found_end
    {
    }; 

    // Custom visitor to interrupt the search when the goal is found
    struct astar_end_visitor : public boost::default_astar_visitor
    {
        astar_end_visitor(BoostGraph::vertex_descriptor goal) : goal_vertex(goal) {}

        void examine_vertex(BoostGraph::vertex_descriptor u, [[maybe_unused]] const BoostGraph& g)
        {
            if (u == goal_vertex) {
                found = true;
                throw found_end();
            }
        }

        BoostGraph::vertex_descriptor goal_vertex;
        bool found = false;
    };

    struct astar_end_visitor_sph_graph : public boost::default_astar_visitor
    {
        astar_end_visitor_sph_graph(boost::graph_traits<GraphView>::vertex_descriptor goal) : goal_vertex(goal) {}

        void examine_vertex(boost::graph_traits<GraphView>::vertex_descriptor u, [[maybe_unused]] const GraphView& g)
        {
            if (u == goal_vertex) {
                found = true;
                throw found_end();
            }
        }

        boost::graph_traits<GraphView>::vertex_descriptor goal_vertex;
        bool found = false;
    };

    BoostGraph createBoostGraph(const GraphBaseInterface& graphView)
    {
        Log::info("createBoostGraph: create boost graph from {0} points with {1} connections and {2} edges", graphView.getNumPoints(), graphView.getKnnDistances().size(), graphView.getKnnDistances().size() - graphView.getNumPoints());

        BoostGraph bgraph;
        utils::ProgressBar progress(graphView.getNumPoints());

        for (int64_t point = 0; point < graphView.getNumPoints(); point++)
        {

            for (int64_t neighborLocal = 0; neighborLocal < graphView.getK(point); neighborLocal++)
            {
                const auto [neighborGlobal, distance] = graphView.getNeighborDistanceN(point, neighborLocal);
                if (point != neighborGlobal)
                    boost::add_edge(point, neighborGlobal, distance, bgraph);

            }

            progress.update();
        }
        progress.finish();

        //auto v = boost::num_vertices(bgraph);
        //auto e = boost::num_edges(bgraph);

        //for (auto ep = boost::edges(bgraph); ep.first != ep.second; ++ep.first)
        //    std::cout << "Edge: " << boost::source(*ep.first, bgraph) << " -> " << boost::target(*ep.first, bgraph) << std::endl;

        assert(boost::num_vertices(bgraph) == static_cast<size_t>(graphView.getNumPoints()));
        assert(boost::num_edges(bgraph) == graphView.getKnnDistances().size() - static_cast<size_t>(graphView.getNumPoints()));

        Log::info("createBoostGraph: boost graph has {0} points and {1} edges", boost::num_vertices(bgraph), boost::num_edges(bgraph));

        return bgraph;
    }

    static void getDistanceAndPath([[maybe_unused]] int64_t startID, int64_t endID, const std::vector<float>& distance_map, const std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist) {
        // Extract shortest path
        for (uint64_t v = endID; /* no checks*/; v = predecessor_map[v])
        {
            geodesic.emplace(geodesic.begin(), v);
            if (predecessor_map[v] == v)
                break;
        }

        g_dist = distance_map[endID];

        assert(g_dist >= 0.f);
        assert(geodesic.size() > 2);
        assert(geodesic[0] == startID);
        assert(geodesic[geodesic.size() - 1] == endID);
    }


    void astarBoost(const BoostGraph& bgraph, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist)
    {
        // Return -1 geodesic distance if no path was found
        geodesic.clear();
        g_dist = -1.f;

#ifndef IS_ARM64
        // on ARM this seems to result in std::is_same<unsigned long, unsigned long long>
        static_assert(std::is_same<boost::property_traits<boost::property_map<BoostGraph, boost::vertex_index_t>::type>::value_type, uint64_t>::value,
            "BoostGraph::vertex_descriptor is not uint64_t.");
#endif

        // Define a distance map to store the shortest distance from the start vertex to each 
        distance_map.clear();
        distance_map.resize(boost::num_vertices(bgraph));

        // Define a predecessor map to store the previous vertex from each vertex
        predecessor_map.clear();
        predecessor_map.resize(boost::num_vertices(bgraph));

        try
        {
            // Define some A* help structures
            astar_end_visitor end_vis(endID);
            euclidean_heuristic heuristic(endID, data.getData(), data.getNumDimensions());
            auto d_map = boost::make_iterator_property_map(distance_map.begin(), boost::get(boost::vertex_index, bgraph));
            auto p_map = boost::predecessor_map(boost::make_iterator_property_map(predecessor_map.begin(), boost::get(boost::vertex_index, bgraph)));
            auto w_map = boost::get(boost::edge_weight, bgraph);

            // Compute shortest path
            boost::astar_search(bgraph, startID, heuristic,
                p_map
                .distance_map(d_map)
                .weight_map(w_map)
                .visitor(end_vis));
        }
        catch ([[maybe_unused]] const found_end& fg)
        {
            getDistanceAndPath(startID, endID, distance_map, predecessor_map, geodesic, g_dist);
        }


    }

    void astarBoost(const GraphView& graphView, const DataView& data, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist)
    {
        // Return -1 geodesic distance if no path was found
        geodesic.clear();
        g_dist = -1.f;

#ifndef IS_ARM64
        // on ARM this seems to result in std::is_same<unsigned long, unsigned long long>
        static_assert(std::is_same<boost::property_traits<boost::property_map<BoostGraph, boost::vertex_index_t>::type>::value_type, uint64_t>::value,
            "BoostGraph::vertex_descriptor is not uint64_t.");
#endif

        // Define a distance map to store the shortest distance from the start vertex to each 
        distance_map.clear();
        distance_map.resize(boost::num_vertices(graphView));

        // Define a predecessor map to store the previous vertex from each vertex
        predecessor_map.clear();
        predecessor_map.resize(boost::num_vertices(graphView));

        try
        {
            // Define some A* help structures
            astar_end_visitor_sph_graph end_vis(endID);
            euclidean_heuristic_sph_graph heuristic(endID, data.getData(), data.getNumDimensions());
            auto d_map = boost::make_iterator_property_map(distance_map.begin(), boost::get(boost::vertex_index, graphView));
            auto p_map = boost::predecessor_map(boost::make_iterator_property_map(predecessor_map.begin(), boost::get(boost::vertex_index, graphView)));
            auto w_map = boost::get(boost::edge_weight, graphView);

            // Compute shortest path
            boost::astar_search(graphView, startID, heuristic,
                p_map
                .distance_map(d_map)
                .weight_map(w_map)
                .visitor(end_vis));
        }
        catch ([[maybe_unused]] const found_end& fg)
        {
            getDistanceAndPath(startID, endID, distance_map, predecessor_map, geodesic, g_dist);
        }


    }

    void dijkstraBoost(const BoostGraph& bgraph, int64_t startID, int64_t endID, std::vector<float>& distance_map, std::vector<uint64_t>& predecessor_map, std::vector<int64_t>& geodesic, float& g_dist)
    {
        // Return -1 geodesic distance if no path was found
        geodesic.clear();
        g_dist = -1.f;

        // Define some Dijkstra help structures
        auto d_map = boost::make_iterator_property_map(distance_map.begin(), get(boost::vertex_index, bgraph));
        auto p_map = boost::make_iterator_property_map(predecessor_map.begin(), get(boost::vertex_index, bgraph));

        // Compute Dijkstra
        boost::dijkstra_shortest_paths(bgraph, startID, boost::distance_map(d_map).predecessor_map(p_map));

        if (distance_map[endID] != std::numeric_limits<float>::max())
            getDistanceAndPath(startID, endID, distance_map, predecessor_map, geodesic, g_dist);

    }


} // namespace sph::utils
