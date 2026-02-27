#include "Graph.hpp"

#include "CommonDefinitions.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <cassert>

#if SPH_DEBUG
    #include <numeric>
#endif

namespace sph::utils
{

    /// ////////////////// ///
    /// GraphBaseInterface ///
    /// ////////////////// ///

    bool GraphBaseInterface::hasUniqueNeighbors() const
    {
        const auto np = getNumPoints();
        bool res = true;

        for (int64_t p = 0; p < static_cast<int64_t>(np); ++p)
        {
            const auto neighbors = getNeighbors(p);
            auto it = std::adjacent_find(neighbors.begin(), neighbors.end());
            if (it != neighbors.end())
            {
                res = false;
                break;
            }
        }

        return res;
    }


    /// ///////////// ///
    /// GraphBaseData ///
    /// ///////////// ///

    GraphBaseData::GraphBaseData(const GraphBaseData& graphBaseData) :
        knnIndices(graphBaseData.knnIndices),
        knnDistances(graphBaseData.knnDistances),
        numPoints(graphBaseData.numPoints),
        symmetric(graphBaseData.symmetric)
    {
    }

    GraphBaseData::GraphBaseData(GraphBaseData&& graphBaseData) noexcept :
        GraphBaseData(std::move(graphBaseData.knnIndices), std::move(graphBaseData.knnDistances), graphBaseData.numPoints, graphBaseData.symmetric)
    {
        symmetric = graphBaseData.symmetric;
    }

    GraphBaseData::GraphBaseData(std::vector<int64_t>&& idx, std::vector<float>&& dists, int64_t numPoints, bool symmetric) noexcept :
        knnIndices(std::move(idx)),
        knnDistances(std::move(dists)),
        numPoints(numPoints),
        symmetric(symmetric)
    {
        assert(knnIndices.size() == knnDistances.size());
    }

    /// ///////////// ///
    /// GraphBaseView ///
    /// ///////////// ///

    GraphBaseView::GraphBaseView(const GraphBaseView& graphBaseView) :
        knnIndices(graphBaseView.knnIndices),
        knnDistances(graphBaseView.knnDistances),
        numPoints(graphBaseView.numPoints),
        symmetric(graphBaseView.symmetric)
    {
    }

    GraphBaseView::GraphBaseView(std::vector<int64_t>* idx, std::vector<float>* dists, int64_t* numPoints, bool* symmetric) :
        knnIndices(idx),
        knnDistances(dists),
        numPoints(numPoints),
        symmetric(symmetric)
    {
        assert(knnIndices->size() == knnDistances->size());
    }

    /// ///// ///
    /// Graph ///
    /// ///// ///


    Graph::Graph(const Graph& graph) :
        GraphBaseData(graph)
    {
        nns = graph.nns;
        offsets = graph.offsets;
    }

    Graph& Graph::operator= (const Graph& graph)
    {
        if (this != &graph) {  // Self-assignment check
            knnIndices = graph.knnIndices;
            knnDistances = graph.knnDistances;
            symmetric = graph.symmetric;
            numPoints = graph.numPoints;
            nns = graph.nns;
            offsets = graph.offsets;
        }
        return *this;
    }

    Graph::Graph(Graph&& graph) noexcept :
        Graph(std::move(graph.knnIndices), std::move(graph.knnDistances), std::move(graph.nns), std::move(graph.offsets), graph.symmetric)
    {
    }

    Graph::Graph(std::vector<int64_t>&& idx, std::vector<float>&& dists, std::vector<int64_t>&& numNeighbors, std::vector<int64_t>&& offset, bool symmetric) noexcept :
        GraphBaseData(std::move(idx), std::move(dists), numNeighbors.size(), symmetric),
        nns(std::move(numNeighbors)),
        offsets(std::move(offset))
    {
        assert(static_cast<size_t>(std::accumulate(nns.begin(), nns.end(), 0ll)) == knnIndices.size());
    }

    int64_t Graph::isDirectNeighbor(uint64_t id1, uint64_t id2) const
    {
        const auto numNeighs = static_cast<uint64_t>(nns[id1]);
        for (uint64_t neighID = 0; neighID < numNeighs; neighID++) {
            if (getNeighborN(id1, neighID) == static_cast<int64_t>(id2))
            {
                return neighID;
            }
        }
        return -1;
    }

    GraphView Graph::getGraphView()
    {
        return GraphView{ &knnIndices, &knnDistances, &numPoints, &nns, &offsets, &symmetric };
    }

    const GraphView Graph::getGraphView() const
    {
        return GraphView{ const_cast<vi64*>(&knnIndices), const_cast<vf32*>(&knnDistances), const_cast<int64_t*>(&numPoints), const_cast<vi64*>(&nns), const_cast<vi64*>(&offsets), const_cast<bool*>(&symmetric) };
    }

    /// ///////// ///
    /// GraphView ///
    /// ///////// ///

    GraphView::GraphView(const GraphView& graphView) :
        GraphBaseView(graphView)
    {
        nns = graphView.nns;
        offsets = graphView.offsets;
    }

    GraphView& GraphView::operator= (const GraphView& graphView)
    {
        if (this != &graphView) {  // Self-assignment check
            knnIndices = graphView.knnIndices;
            knnDistances = graphView.knnDistances;
            numPoints = graphView.numPoints;
            nns = graphView.nns;
            offsets = graphView.offsets;
            symmetric = graphView.symmetric;
        }
        return *this;
    }

    GraphView::GraphView(std::vector<int64_t>* idx, std::vector<float>* dists, int64_t* numPoints, std::vector<int64_t>* ns, std::vector<int64_t>* offset, bool* symmetric) :
        GraphBaseView(idx, dists, numPoints, symmetric),
        nns(ns),
        offsets(offset)
    {
        assert(static_cast<size_t>(std::accumulate(nns->begin(), nns->end(), 0ll)) == idx->size());
    }

    int64_t GraphView::isDirectNeighbor(uint64_t id1, uint64_t id2) const
    {
        const auto numNeighs = (*nns)[id1];
        for (int64_t neighID = 0; neighID < numNeighs; neighID++) {
            if (getNeighborN(id1, neighID) == static_cast<int64_t>(id2))
            {
                return neighID;
            }
        }
        return -1;
    }

    /// ////// ///
    /// KGraph ///
    /// ////// ///

    KGraph::KGraph(const KGraph& kGraph) : 
        GraphBaseData(kGraph),
        k(kGraph.k)
    {
    }

    KGraph& KGraph::operator= (const KGraph& kGraph)
    {
        if (this != &kGraph) {  // Self-assignment check
            knnIndices = kGraph.knnIndices;
            knnDistances = kGraph.knnDistances;
            numPoints = kGraph.numPoints;
            k = kGraph.k;
        }
        return *this;
    }

    KGraph::KGraph(KGraph&& kGraph) noexcept :
        KGraph(std::move(kGraph.knnIndices), std::move(kGraph.knnDistances), kGraph.k)
    {

    }

    KGraph::KGraph(std::vector<int64_t>&& idx, std::vector<float>&& dists, int64_t k) noexcept :
        GraphBaseData(std::move(idx), std::move(dists), idx.size() / k, false),
        k(k)
    {
        assert(k > 0);
        if (k <= 0)
            Log::error("KGraph::KGraph: k cannot not be <= 0");

        assert(knnIndices.size() == static_cast<size_t>(k * numPoints));
    }

    int64_t KGraph::isDirectNeighbor(uint64_t id1, uint64_t id2) const
    {
        //auto neighbors = getNeighbors(id);
        //auto it = std::find(neighbors.begin(), neighbors.end(), n);
        //if (it != neighbors.end()) {
        //    return std::distance(neighbors.begin(), it);
        //}
        //return -1;

        for (int64_t neighID = 0; neighID < k; neighID++) {
            if (getNeighborN(id1, neighID) == static_cast<int64_t>(id2))
            {
                return neighID;
            }
        }
        return -1;
    }

    KGraphView KGraph::getKGraphView()
    {
        return KGraphView{ &knnIndices, &knnDistances, &numPoints, &k };
    }

    const KGraphView KGraph::getKGraphView() const
    {
        return KGraphView{ const_cast<vi64*>(&knnIndices), const_cast<vf32*>(&knnDistances), const_cast<int64_t*>(&numPoints), const_cast<int64_t*>(&k) };
    }

    /// ////////// ///
    /// KGraphView ///
    /// ////////// ///

    KGraphView::KGraphView(const KGraphView& kGraphView) :
        GraphBaseView(kGraphView),
        k(kGraphView.k)
    {
    }

    KGraphView& KGraphView::operator= (const KGraphView& kGraphView)
    {
        if (this != &kGraphView) {  // Self-assignment check
            knnIndices = kGraphView.knnIndices;
            knnDistances = kGraphView.knnDistances;
            numPoints = kGraphView.numPoints;
            k = kGraphView.k;
        }
        return *this;
    }

    KGraphView::KGraphView(std::vector<int64_t>*idx, std::vector<float>*dists, int64_t* numPoints, int64_t* knn) :
        GraphBaseView(idx, dists, numPoints, nullptr),
        k(knn)
    {
    }

    int64_t KGraphView::isDirectNeighbor(uint64_t id1, uint64_t id2) const
    {
        const auto numNeighs = static_cast<uint64_t>(*k);
        for (uint64_t neighID = 0; neighID < numNeighs; neighID++) {
            if (getNeighborN(id1, neighID) == static_cast<int64_t>(id2))
            {
                return neighID;
            }
        }
        return -1;
    }

    /// ///////// ///
    /// Utilities ///
    /// ///////// ///

    // Append a new vertex and init the knn neighbor IDs
    void appendNode(Graph& g, std::vector<uint64_t>&& newNeighborIDs)
    {
        g.getNns().push_back(newNeighborIDs.size() + 1);

        if (g.getNns().size() > 1)
            g.getOffsets().push_back(g.getOffsets().back() + g.getNns().at(g.getNns().size() - 2));
        else
            g.getOffsets().push_back(0);

        auto& knnIndices = g.getKnnIndices();
        knnIndices.push_back(g.getNns().size() - 1);
        knnIndices.insert(knnIndices.end(), std::make_move_iterator(newNeighborIDs.begin()), std::make_move_iterator(newNeighborIDs.end()));
        g.getKnnDistances().resize(knnIndices.size(), 0.f);
    }

    // Append a new vertex and init the knn neighbor IDs and distances
    void appendNode(Graph& g, std::vector<uint64_t>&& newNeighborIDs, std::vector<float>&& newNeighborDistances)
    {
        g.getNns().push_back(newNeighborIDs.size() + 1);

        if (g.getNns().size() > 1)
            g.getOffsets().push_back(g.getOffsets().back() + g.getNns().at(g.getNns().size() - 2));
        else
            g.getOffsets().push_back(0);

        auto& knnIndices = g.getKnnIndices();
        auto& knnDistances = g.getKnnDistances();
        knnIndices.push_back(g.getNns().size() - 1);
        knnIndices.insert(knnIndices.end(), std::make_move_iterator(newNeighborIDs.begin()), std::make_move_iterator(newNeighborIDs.end()));
        knnDistances.push_back(0.f);
        knnDistances.insert(knnDistances.end(), std::make_move_iterator(newNeighborDistances.begin()), std::make_move_iterator(newNeighborDistances.end()));
    }


} // namespace sph::utils
