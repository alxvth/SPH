#pragma once

#include <cstdint>

#include "Graph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/graph_selectors.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/pending/property.hpp>

namespace sph::utils
{
    struct DataView;

    // Define the Boost graph type with directed edges
    using BoostGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
        boost::no_property, boost::property<boost::edge_weight_t, float>,       // vertex distances are in float
        boost::property<boost::vertex_index_t, uint64_t>>;                      // this sets BoostGraph::vertex_descriptor to uint64_t
    using BoostEdge = boost::graph_traits<BoostGraph>::edge_descriptor;
    using BoostVertex = boost::graph_traits<BoostGraph>::vertex_descriptor;     // since we use boost::no_property, this won't tell you much

    using BoostGraph32 = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
        boost::no_property, boost::property<boost::edge_weight_t, float>,       // vertex distances are in float
        boost::property<boost::vertex_index_t, uint32_t>>;                      // this sets BoostGraph::vertex_descriptor to uint32_t
    using BoostEdge32 = boost::graph_traits<BoostGraph32>::edge_descriptor;
    using BoostVertex32 = boost::graph_traits<BoostGraph32>::vertex_descriptor; // since we use boost::no_property, this won't tell you much

    using BoostGraphFull = boost::adjacency_matrix<boost::directedS,
        boost::no_property, boost::property<boost::edge_weight_t, float>>;      // vertex distances are in float
    using BoostEdgeFull = boost::graph_traits<BoostGraphFull>::edge_descriptor;
    using BoostVertexFull = boost::graph_traits<BoostGraphFull>::vertex_descriptor;

    using BoostGraphFullU = boost::adjacency_matrix<boost::undirectedS,         // undirected graph
        boost::no_property, boost::property<boost::edge_weight_t, float>>;      // vertex distances are in float
    using BoostEdgeFullU = boost::graph_traits<BoostGraphFullU>::edge_descriptor;
    using BoostVertexFullU = boost::graph_traits<BoostGraphFullU>::vertex_descriptor;

    BoostGraph createBoostGraph(const GraphBaseInterface& graphView);

}
