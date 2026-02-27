#pragma once

#include "CommonDefinitions.hpp"

#include "Graph.hpp"
#include "GraphBoost.hpp"

namespace sph::utils
{
    // returns number of adjusted points
    int64_t ensureClosestPointIsSelf(GraphBaseInterface* graphView, bool verbose = true);

    // symmetrizes a graph
    Graph symmetrizeGraphOld(const GraphBaseInterface& graphView, bool verbose = true);     // takes min of duplicate entries
    Graph symmetrizeGraph(const GraphBaseInterface& graphView, bool verbose = true);        // takes min of duplicate entries
    Graph symmetrizeGraphEigen(const GraphBaseInterface& graphView, bool verbose = true);   // computes (mat + mat) * 0.5

    // labels weakly connected components in a directed graph
    // computes symmetric graph and calls labelGraphStrongComponents
    int64_t labelGraphWeakComponents(const BoostGraph& bgraph, sph::vi64& labels);
    int64_t labelGraphWeakComponents(const GraphView& graphView, sph::vi64& labels);
    inline int64_t labelGraphWeakComponents(const GraphView& graphView)
    {
        sph::vi64 labels;
        return labelGraphWeakComponents(graphView, labels);
    }

    // labels strongly connected components in a directed graph
    int64_t labelGraphStrongComponents(const BoostGraph& bgraph, sph::vi64& labels);
    int64_t labelGraphStrongComponents(const GraphView& graphView, sph::vi64& labels);
    inline int64_t labelGraphStrongComponents(const GraphView& graphView)
    {
        sph::vi64 labels;
        return labelGraphStrongComponents(graphView, labels);
    }

}
