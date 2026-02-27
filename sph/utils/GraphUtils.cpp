#include "GraphUtils.hpp"

#include "Algorithms.hpp"
#include "GraphAdapterBoost.hpp"
#include "Logger.hpp"
#include "ProgressBar.hpp"
#include "SparseMatrixAlgorithms.hpp"

#include <ankerl/unordered_dense.h>
#include <boost/graph/strong_components.hpp>
#include <Eigen/SparseCore>     // Eigen::SparseVector
#include <range/v3/view/zip.hpp>

#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>

namespace sph::utils
{
    using namespace sph;

    int64_t ensureClosestPointIsSelf(GraphBaseInterface* graphView, bool verbose)
    {
        auto allDistancesAreZero = [graphView](int64_t p) -> bool {
            const auto dists = graphView->getDistances(p);
            return std::all_of(dists.begin(), dists.end(), [](const auto val) { return val == 0; });
            };

        const auto numPoints = graphView->getNumPoints();

        utils::ProgressBar progress(numPoints, verbose);
        int64_t numAdjustedPoints = 0;

        SPH_PARALLEL
        for (int64_t p = 0; p < numPoints; ++p) {
            if (graphView->getNeighborN(p, 0) != p) {
                Log::trace("ensureClosestPointIsSelf: The first nn should be the point itself: swap it (Point {0})", p);

                auto neighbors = graphView->getNeighborsRef(p);
                auto distances = graphView->getDistancesRef(p);

                assert(neighbors.size() == distances.size());

                size_t swap_pos = neighbors.size() - 1;

                auto it = std::find(neighbors.begin(), neighbors.end(), p);
                if (it != neighbors.end())
                {
                    swap_pos = std::distance(neighbors.begin(), it);
                    assert(swap_pos > 0);
                }
                else if (allDistancesAreZero(p))
                {
                    // in case all knn distances are zero, the point itself might
                    // not be among its own knn. (Arbitrarily) remove the last knn
                    neighbors[swap_pos] = p;
                }
                else
                {
                    Log::debug("ensureClosestPointIsSelf: p {0} is not among its neighbors. Removing most distance neighbor in its favor.", p);

                    // remove the most distance neighbor by shifting all neighbors
                    // to the right and inserting the point at first position
                    for (std::size_t i = neighbors.size() - 1; i > 0; --i) {
                        neighbors[i] = neighbors[i - 1];
                        distances[i] = distances[i - 1];
                    }

                    swap_pos = 0;
                    neighbors[0] = p;
                    distances[0] = 0;
                }

                assert(neighbors[swap_pos] == p);
                assert(distances[swap_pos] == 0);

                std::swap(neighbors[0], neighbors[swap_pos]);
                std::swap(distances[0], distances[swap_pos]);

                assert(graphView->getNeighborN(p, 0) == p);
                assert(graphView->getDistanceN(p, 0) == 0);

                numAdjustedPoints++;
            }

            SPH_PARALLEL_CRITICAL
            progress.update();
        }
        progress.finish();

        assert(graphView->isValid());

        return numAdjustedPoints;
    }

    Graph symmetrizeGraphOld(const GraphBaseInterface& graphView, bool verbose)
    {
        using hash = ankerl::unordered_dense::hash<int64_t>;
        using hashmap = ankerl::unordered_dense::map<int64_t, float, hash>;

        Graph graph;

        const auto np = graphView.getNumPoints();
        const auto k = graphView.getK();

        graph.numPoints = np;

        vvi64 symIndex(np);
        vvf32 symDists(np);

        // Conservatively reserve some space
        for (int64_t i = 0; i < np; i++)
        {
            symIndex[i].reserve(k);
            symDists[i].reserve(k);
        }

        utils::ProgressBar progress(np * 2, verbose);

        // Copy knns and make symmetric
        for (int64_t i = 0; i < np; i++)
        {
            const auto index = graphView.getNeighbors(i);
            const auto dists = graphView.getDistances(i);

            assert(dists.size() == index.size());

            //for (int64_t j = 0; j < index.size(); j++)
            for (const auto& [id, dist] : ranges::views::zip(index, dists))
            {
                symIndex[i].push_back(id);
                symIndex[id].push_back(i);

                symDists[i].push_back(dist);
                symDists[id].push_back(dist);
            }

            progress.update();
        }

        progress.update(np);

        // Sort neighbors and distances
        for (int64_t i = 0; i < np; i++)
        {
            // Keep only single entries for each neighbor, those with smaller distance
            hashmap unique_map;
            for (const auto& [id, dist] : ranges::views::zip(symIndex[i], symDists[i]))
            {
                if (unique_map.contains(id) && unique_map[id] < dist)
                    continue;

                unique_map[id] = dist;
            }

            // Save some memory
            symIndex[i].clear();
            symDists[i].clear();

            // Sort the neighbors-distance pairs
            std::vector<std::pair<int64_t, float>> pairs(unique_map.begin(), unique_map.end());
            std::sort(SPH_PARALLEL_EXECUTION
                pairs.begin(), 
                pairs.end(), 
                [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {
                if (a.second == b.second)
                    return a.first < b.first;
                return a.second < b.second;
                });

            unique_map.clear();

            // Append the neighbors-distance pairs to the graph structure
            for (const auto& pair : pairs) {
                graph.getKnnIndices().push_back(pair.first);
                graph.getKnnDistances().push_back(pair.second);
            }

            graph.getNns().push_back(pairs.size());
            graph.updateOffsets();

            progress.update();
        }

        progress.finish();

        graph.symmetric = true;

        ensureClosestPointIsSelf(&graph, verbose);

        assert(graph.isValid());

        return graph;
    }

    template<typename INT, typename = std::enable_if_t<std::is_same_v<INT, int32_t> || std::is_same_v<INT, int64_t>> >
    static Graph symmetrizeGraphImpl(const GraphBaseInterface& graphView, bool verbose)
    {
        using hash = ankerl::unordered_dense::hash<INT>;
        using hashmap = ankerl::unordered_dense::map<INT, float, hash>;

        const auto np = graphView.getNumPoints();

        assert(np < std::numeric_limits<INT>::max());

        // This entire approach is rather memory consuming...
        std::vector<hashmap> symPairs(np);

        utils::ProgressBar progress(np * 3, verbose);

        {
            std::vector< std::vector<std::pair<INT, float>>> emptyVals(np);

            SPH_PARALLEL
            for (INT i = 0; i < np; i++)
            {
                const auto index = graphView.getNeighbors(i);
                const auto dists = graphView.getDistances(i);

                assert(dists.size() == index.size());

                auto& symPairsI = symPairs[i];
                symPairsI.reserve(index.size());

                auto& emptyValI = emptyVals[i];
                emptyValI.reserve(index.size() / 4);

                for (const auto& [j, dist] : ranges::views::zip(index, dists)) {
                    float value = dist;

                    const int64_t directNeighbor = graphView.isDirectNeighbor(j, i);

                    if (directNeighbor == -1)
                        emptyValI.push_back({ static_cast<INT>(j) , value });
                    else
                        value = std::min(value, graphView.getDistanceN(j, directNeighbor));

                    symPairsI.emplace(static_cast<INT>(j), value);
                }

                progress.update();
            }

            for (INT i = 0; i < np; i++)
            {
                auto& emptyValI = emptyVals[i];

                for (const auto& [j, dist] : emptyValI)
                    symPairs[j].emplace(i, dist);

                emptyValI.clear();
                emptyValI = {};
                progress.update();
            }
        }

        Graph graph;
        graph.numPoints = np;

        auto& graphIndices = graph.getKnnIndices();
        auto& graphDistances = graph.getKnnDistances();

        graphIndices.reserve(graphView.getKnnIndices().size());
        graphDistances.reserve(graphView.getKnnDistances().size());

        // Sort neighbors and distances
        for (int64_t i = 0; i < np; i++)
        {
            auto& symPairsI = symPairs[i];

            std::vector<std::pair<INT, float>> currentPairs;
            currentPairs.reserve(symPairsI.size());
            for (const auto& [key, value] : symPairsI) {
                currentPairs.emplace_back(key, value);
            }

            symPairsI.clear();
            symPairsI = {};

            // Sort the neighbors-distance pairs and keep unique
            std::sort(SPH_PARALLEL_EXECUTION
                currentPairs.begin(), currentPairs.end(), [](const auto& a, const auto& b) {
                    if (a.second == b.second)
                        return a.first < b.first;
                    return a.second < b.second;
                });

            // Append the neighbors-distance pairs to the graph structure
            for (const std::pair<INT, float>& pair : currentPairs) {
                graphIndices.push_back(pair.first);
                graphDistances.push_back(pair.second);
            }

            graph.getNns().push_back(currentPairs.size());
            progress.update();
        }

        progress.finish();

        graph.updateOffsets();
        graph.symmetric = true;

        ensureClosestPointIsSelf(&graph, verbose);

        assert(graph.isValid());

        return graph;
    }

    Graph symmetrizeGraph(const GraphBaseInterface& graphView, bool verbose)
    {
        if (graphView.getNumPoints() < std::numeric_limits<int32_t>::max())
            return symmetrizeGraphImpl<int32_t>(graphView, verbose);

        return symmetrizeGraphImpl<int64_t>(graphView, verbose);
    }

    Graph symmetrizeGraphEigen(const GraphBaseInterface& graphView, bool verbose)
    {
        Eigen::SparseMatrix<float, Eigen::RowMajor, int> symmetric_mat;
        {
            Eigen::SparseMatrix<float, Eigen::RowMajor, int> mat = createSparseMatrixFromGraph(graphView);
            Eigen::SparseMatrix<float, Eigen::RowMajor, int> transpose = mat.transpose();
            symmetric_mat = (mat + transpose) * 0.5f;
            symmetric_mat.makeCompressed();
        }

        assert(static_cast<int64_t>(symmetric_mat.rows()) == graphView.getNumPoints());
        assert(static_cast<int64_t>(symmetric_mat.cols()) == graphView.getNumPoints());

        Graph graph;
        graph.numPoints = graphView.getNumPoints();

        for (size_t row = 0; row < static_cast<size_t>(symmetric_mat.rows()); ++row) {
            std::vector<std::pair<int64_t, float>> pairs;
            pairs.push_back({ row, 0.f});

            for (Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator it(symmetric_mat, row); it; ++it)
                pairs.push_back({ it.col(), it.value() });

            // Sort the neighbors-distance pairs
            std::sort(SPH_PARALLEL_EXECUTION
                pairs.begin(), 
                pairs.end(), 
                [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {
                if (a.second == b.second)
                    return a.first < b.first;
                return a.second < b.second;
                });

            // Append the neighbors-distance pairs to the graph structure
            for (const auto& pair : pairs) {
                graph.getKnnIndices().push_back(pair.first);
                graph.getKnnDistances().push_back(pair.second);
            }

            graph.getNns().push_back(pairs.size());
        }

        graph.updateOffsets();
        graph.symmetric = true;

        ensureClosestPointIsSelf(&graph, verbose);

        assert(graph.isValid());

        return graph;
    }

    int64_t labelGraphWeakComponents(const utils::BoostGraph& bgraph, vi64& labels)
    {
        // create symmetric graph
        auto numVertices = boost::num_vertices(bgraph);
        utils::BoostGraph symmetricGraph = bgraph;
        for (auto [ei, ei_end] = boost::edges(symmetricGraph); ei != ei_end; ++ei) {
            BoostVertex u = boost::source(*ei, symmetricGraph);
            BoostVertex v = boost::target(*ei, symmetricGraph);

            if (!boost::edge(v, u, symmetricGraph).second) {
                boost::add_edge(v, u, symmetricGraph);
            }
        }

        assert(numVertices == boost::num_vertices(symmetricGraph));

        labels.resize(numVertices);

        int64_t num_components = boost::strong_components(symmetricGraph, &labels[0]);

        return num_components;
    }

    int64_t labelGraphWeakComponents(const GraphView& graphView, vi64& labels)
    {
        int64_t num_components = -1;

        if (!graphView.isSymmetric())
        {
            utils::Graph symmetricGraph = utils::symmetrizeGraph(graphView);
            num_components = labelGraphStrongComponents(symmetricGraph.getGraphView(), labels);
        }
        else
            num_components = labelGraphStrongComponents(graphView, labels);

        return num_components;
    }

    int64_t labelGraphStrongComponents(const utils::BoostGraph& bgraph, vi64& labels)
    {
        labels.resize(boost::num_vertices(bgraph));
        int64_t num_components = boost::strong_components(bgraph, &labels[0]);

        return num_components;
    }

    int64_t labelGraphStrongComponents(const GraphView& graphView, vi64& labels)
    {
        labels.resize(boost::num_vertices(graphView));
        int64_t num_components = boost::strong_components(graphView, &labels[0]);

        return num_components;
    }

} // namespace sph::utils

