#pragma once

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "Graph.hpp"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/iterator/iterator_categories.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/property_map/property_map.hpp>

// https://www.boost.org/doc/libs/1_86_0/libs/graph/doc/graph_concepts.html
// https://www.boost.org/doc/libs/1_86_0/boost/graph/leda_graph.hpp and https://www.boost.org/doc/libs/1_86_0/libs/graph/doc/leda_conversion.html
// dijkstra_shortest_paths: Vertex List Graph and Incidence Graph
// astar_search: Vertex List Graph and Incidence Graph
// strong_components: Vertex List Graph and Incidence Graph

namespace boost
{
    struct GraphBoostTags :
        public virtual boost::incidence_graph_tag,
        public virtual boost::vertex_list_graph_tag
    {
    };

    // GraphView //

    template <>
    struct graph_traits<sph::utils::GraphView> {
        // Graph //
        typedef int64_t vertex_descriptor;
        typedef std::pair<vertex_descriptor, vertex_descriptor> edge_descriptor;
        typedef directed_tag directed_category;
        typedef disallow_parallel_edge_tag edge_parallel_category;

        typedef GraphBoostTags traversal_category;
        //typedef vertex_list_graph_tag vertex_list_category; // maybe not needed?

        static vertex_descriptor null_vertex()
        {
            return -1;
        }

        // IncidenceGraph //

        class GraphViewOutEdgeIterator :
            public boost::iterator_facade <
            GraphViewOutEdgeIterator,
            graph_traits<sph::utils::GraphView>::edge_descriptor,  // (source, target)
            boost::forward_traversal_tag,
            graph_traits<sph::utils::GraphView>::edge_descriptor
            >
        {
        public:
            GraphViewOutEdgeIterator() : m_graph(nullptr), m_vertex(-1), m_index(-1) {}
            GraphViewOutEdgeIterator(const sph::utils::GraphView* g, int64_t v, int64_t i) :
                m_graph(g),
                m_vertex(v),
                m_index(i)
            {}

        private:
            friend class boost::iterator_core_access;

            void increment() {
                ++m_index;
                assert(m_index <= m_graph->getNns()[m_vertex]);
            }
            void decrement() {
                --m_index;
                assert(m_index > 0);
            }
            bool equal(const GraphViewOutEdgeIterator& other) const {
                //return m_graph == other.m_graph && m_vertex == other.m_vertex && m_index == other.m_index;
                return m_vertex == other.m_vertex && m_index == other.m_index;
            }
            std::pair<int64_t, int64_t> dereference() const {
                return { m_vertex, m_graph->getNeighborN(m_vertex, m_index) };
            }

            const sph::utils::GraphView* m_graph;
            int64_t m_vertex;
            int64_t m_index;
        };

        typedef GraphViewOutEdgeIterator out_edge_iterator;
        typedef int64_t degree_size_type;

        // VertexListGraph //

        class GraphViewVertexIterator
            : public iterator_facade<GraphViewVertexIterator,
            int64_t,
            bidirectional_traversal_tag,
            const int64_t&,
            const int64_t*>
        {
        public:
            GraphViewVertexIterator() : m_graph(nullptr), m_vertex(-1) {}
            GraphViewVertexIterator(const sph::utils::GraphView* g, int64_t v) :
                m_graph(g),
                m_vertex(v)
            {}

        private:
            const int64_t& dereference() const {
                return m_vertex;
            }

            bool equal(const GraphViewVertexIterator& other) const
            {
                return m_graph == other.m_graph && m_vertex == other.m_vertex;
                //return m_vertex == other.m_vertex;
            }

            void increment() {
                ++m_vertex;
            }
            void decrement() {
                --m_vertex;
            }

            const sph::utils::GraphView* m_graph;
            int64_t m_vertex;

            friend class iterator_core_access;
        };

        typedef GraphViewVertexIterator vertex_iterator;
        typedef size_t vertices_size_type;

        typedef vertex_descriptor key_type;
    };

    // property maps
    struct GraphViewIdMap : public put_get_helper< int64_t, GraphViewIdMap >
    {
        typedef readable_property_map_tag category;
        typedef int64_t value_type;
        typedef int64_t key_type;
        typedef value_type reference;

        GraphViewIdMap() = default;

        // Identity map: vertex descriptor is the index
        int64_t operator[](const key_type& v) const {
            return v;
        }
    };

    template <>
    struct property_map<sph::utils::GraphView, vertex_index_t> {
        typedef GraphViewIdMap type;
        typedef GraphViewIdMap const_type;
    };

    inline GraphViewIdMap get(vertex_index_t, [[maybe_unused]] sph::utils::GraphView& g)
    {
        return GraphViewIdMap();
    }

    inline GraphViewIdMap get(vertex_index_t, [[maybe_unused]] const sph::utils::GraphView& g)
    {
        return GraphViewIdMap();
    }

    struct GraphViewEdgeWeightMap : public boost::put_get_helper<float, GraphViewEdgeWeightMap> {
        using category = boost::readable_property_map_tag;
        using value_type = float;
        using reference = float;
        using key_type = std::pair<int64_t, int64_t>;

        const sph::utils::GraphView* graph;

        GraphViewEdgeWeightMap(const sph::utils::GraphView* g) : graph(g) {}

        float operator[](const key_type& edge) const {
            return graph->getDistance(edge.first, edge.second);
        }
    };

    template <>
    struct property_map<sph::utils::GraphView, edge_weight_t> {
        using type = GraphViewEdgeWeightMap;
        using const_type = GraphViewEdgeWeightMap;
    };

    inline GraphViewEdgeWeightMap get(edge_weight_t, const sph::utils::GraphView& g) {
        return GraphViewEdgeWeightMap(&g);
    }

} // boost

namespace sph::utils {

    // IncidenceGraph
    inline std::pair<boost::graph_traits<GraphView>::out_edge_iterator, boost::graph_traits<GraphView>::out_edge_iterator>
        out_edges(boost::graph_traits<GraphView>::vertex_descriptor v, const GraphView& g) {
        return {
            boost::graph_traits<GraphView>::out_edge_iterator(&g, v, 1),          // 0 is always the point itself
            boost::graph_traits<GraphView>::out_edge_iterator(&g, v, g.getK(v))
        };
    }

    inline boost::graph_traits<GraphView>::vertex_descriptor
        source(boost::graph_traits<GraphView>::edge_descriptor e, [[maybe_unused]] const GraphView& g) {
        return e.first;
    }

    inline boost::graph_traits<GraphView>::vertex_descriptor
        target(boost::graph_traits<GraphView>::edge_descriptor e, [[maybe_unused]] const GraphView& g) {
        return e.second;
    }

    inline boost::graph_traits<GraphView>::degree_size_type
        out_degree(boost::graph_traits<GraphView>::vertex_descriptor v, const GraphView& g) {
        return g.getK(v) - 1; // first neighbor is always point itself
    }

    // VertexListGraph 
    inline std::pair<boost::graph_traits<GraphView>::vertex_iterator, boost::graph_traits<GraphView>::vertex_iterator>
        vertices(const GraphView& g) {
        return { boost::graph_traits<GraphView>::vertex_iterator(&g, 0),
                 boost::graph_traits<GraphView>::vertex_iterator(&g, g.getNumPoints()) };
    }

    inline boost::graph_traits<GraphView>::vertices_size_type
        num_vertices(const GraphView& g) {
        return g.getNumPoints();
    }


} // namespace sph::utils

namespace boost {
    using sph::utils::out_edges;
    using sph::utils::out_degree;
    using sph::utils::source;
    using sph::utils::target;
    using sph::utils::vertices;
    using sph::utils::num_vertices;
}

