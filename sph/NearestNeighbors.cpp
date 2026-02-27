#include "NearestNeighbors.hpp"

#include "utils/Algorithms.hpp"
#include "utils/Distances.hpp"
#include "utils/FileIO.hpp"
#include "utils/GraphUtils.hpp"
#include "utils/Knn.hpp"
#include "utils/Logger.hpp"
#include "utils/Math.hpp"
#include "utils/PrintHelper.hpp"
#include "utils/ProgressBar.hpp"

#pragma warning(disable:4244)       // faiss internal: conversion from 'faiss::idx_t' to 'int', possible loss of data
#pragma warning(disable:4251)       // faiss internal: std class ... needs to have dll-interface to be used by clients of struct 'faiss::InterruptCallback'
#pragma warning(disable:4068)       // MSVC: unknown pragma
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter" // faiss internal: unused parameters id, nbits, nbits_in
#include <faiss/utils/distances.h>  // faiss::knn_L2sqr
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#pragma GCC diagnostic pop
#pragma warning(default:4068)
#pragma warning(default:4251)
#pragma warning(default:4244)

#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <type_traits>

namespace sph {

    using Path = std::filesystem::path;
    using CMaxf = faiss::CMax<float, int64_t>;
    using CMinf = faiss::CMin<float, int64_t>;

    NearestNeighbors::NearestNeighbors() :
        Cacheable("NearestNeighbors")
    {
    }

    NearestNeighbors::NearestNeighbors(const utils::Data& data, const NearestNeighborsSettings& nns) :
        NearestNeighbors()
    {
        _data = data.getDataView();
        _nns = nns;

        if (_data.getNumDimensions() > std::numeric_limits<int>::max())
            Log::warn("NearestNeighbors:: faiss indices only work with max(int) dimensions");
    }

    NearestNeighbors::NearestNeighbors(const utils::DataView& dataView, const NearestNeighborsSettings& nns) :
        NearestNeighbors()
    {
        _data = dataView;
        _nns = nns;

        if (_data.getNumDimensions() > std::numeric_limits<int>::max())
            Log::warn("NearestNeighbors:: faiss indices only work with max(int) dimensions");
    }

    void NearestNeighbors::setData(const utils::Data& data)
    {
        _data = data.getDataView();
    }

    void NearestNeighbors::setData(const utils::DataView& data)
    {
        _data = data;
    }

    void NearestNeighbors::setMetric(utils::KnnMetric knnMetric)
    {
        _nns.knnMetric = knnMetric;

        switch (_nns.knnMetric)
        {
        case utils::KnnMetric::L2:  _faissMetric = static_cast<int64_t>(faiss::MetricType::METRIC_L2); break;
        case utils::KnnMetric::COSINE: [[fallthrough]];
        case utils::KnnMetric::INNER_PRODUCT:
        {
            _faissMetric = static_cast<int64_t>(faiss::MetricType::METRIC_INNER_PRODUCT); 
            _nns.L2squared = false; 
            break;
        }
        }
    }

    void NearestNeighbors::compute(const std::optional<NearestNeighborsSettings>& nns)
    {
        if (nns.has_value())
            setNearestNeighborsSettings(nns.value());

        _hasNeighbors = false;
        _hasSymmetric = false;
        _hasConnected = false;
        _symGraph = {};
        _connectedGraph = {};
        _connectedComponents = std::make_shared<vi64>();
        _numConnectedComponents = -1;

        Log::info("NearestNeighbors::compute: {0} neighbors with metric {1} and index {2}", _nns.numNearestNeighbors, _nns.knnMetric, _nns.knnIndex);

        auto setupKnnGraph = [](utils::Graph& g, const int64_t nn, const int64_t numPoints) {
            g.knnDistances.clear();
            g.knnIndices.clear();
            g.knnDistances.resize(numPoints * nn, -1.0f);
            g.knnIndices.resize(numPoints * nn, -1);
            g.numPoints = numPoints;

            g.updateFixedNumNeighbors(nn);

            assert(g.isValid());
            };

        if (!loadCache())
        {
            setupKnnGraph(_knnGraph, _nns.numNearestNeighbors, _data.getNumPoints());

            initFaissData();

            switch (_nns.knnIndex) {
            case utils::KnnIndex::BruteForce:   utils::computeButeForce(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);         break;
            case utils::KnnIndex::Flat:         utils::computeIndexFlat(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);         break;
            case utils::KnnIndex::IVFFlat:      utils::computeIndexIVFFlat(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);      break;
            case utils::KnnIndex::HNSW:         utils::computeIndexHNSW(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);         break;
            case utils::KnnIndex::HNSWSQ:       utils::computeIndexHNSWSQ(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);       break;
            case utils::KnnIndex::HNSW_IVFPQ:   utils::computeIndexHNSW_IVFPQ(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);   break;
            default:
                Log::error("NearestNeighbors::compute: metric not implemented");
                break;
            }

            if (!checkAllNeighborsExist())
            {
                Log::warn("NearestNeighbors::compute: not all neighbors found, recompute with IndexFlat (this will be slow)");
                setupKnnGraph(_knnGraph, _nns.numNearestNeighbors, _data.getNumPoints());
                utils::computeIndexFlat(_nns.numNearestNeighbors, static_cast<faiss::MetricType>(_faissMetric), _faissData, _knnGraph);
            }

            assert(_knnGraph.isValid());

            ensureFloatingPointIntegrity();

            Log::info("NearestNeighbors::compute: finished searching");

            if (!_nns.L2squared)
            {
                Log::info("NearestNeighbors::compute: Take square root of squared euclidean distances (yields euclidean distances)");
                squareRootOfAllDistances();
            }

            checkDistancesAreNonDecreasing(_knnGraph);

            // It could be that the point itself is not the nearest one if there are identical points.
            // We want the point itself to be the first one!
            Log::info("NearestNeighbors::compute: ensure closest point is self");
            auto numAdjustedPoints = utils::ensureClosestPointIsSelf(&_knnGraph);

            if (numAdjustedPoints > 0)
                Log::info("NearestNeighbors::ensureClosestPointIsSelf: {} points were adjusted (of {})", numAdjustedPoints, _knnGraph.getNumPoints());

            printKnnGraphInfo("compute:");

            _hasNeighbors = true;

            if (_nns.symmetricNeighbors)
                computeSymmetrizedNnGraph();

            if (_nns.computeConnectComponents)
                computeConnectedComponents();

            if (_nns.neighborConnectComponents)
                connectComponents();

            writeCache();
        }

        Log::info("NearestNeighbors::compute: done");
    }

    static void printGraphInfo(const std::string& caller, const std::string& graphName, const utils::GraphView& graphView, const utils::DataView& dataView) {
        Log::info("NearestNeighbors::{}: {} with {} points, {} connections and {} edges", caller, graphView.getNumPoints(), graphName, graphView.getKnnDistances().size(), graphView.getNumEdges());
        Log::info("NearestNeighbors::{} Neighborhood graph sparsity: {:.8f}%", caller, 100. - static_cast<double>(graphView.getKnnIndices().size()) / (dataView.getNumPoints() * dataView.getNumPoints()) * 100.);
    }

    void NearestNeighbors::printKnnGraphInfo(const std::string& caller) {
        printGraphInfo(caller, "Neighborhood graph", _knnGraph.getGraphView(), _data);
    }

    void NearestNeighbors::printSymGraphInfo(const std::string& caller) {
        printGraphInfo(caller, "Symmetric graph", _symGraph.getGraphView(), _data);
    }

    void NearestNeighbors::printConGraphInfo(const std::string& caller) {
        printGraphInfo(caller, "Connected graph", _connectedGraph.getGraphView(), _data);
    }

    void NearestNeighbors::initFaissData()
    {
        if (_nns.knnMetric == utils::KnnMetric::COSINE) {
            // use inner product on L2-normed vectors
            _dataNormed.assign(_data.getData().begin(), _data.getData().end());
            Log::info("NearestNeighbors::init: Normalize data with L2-renormalization");
            faiss::fvec_renorm_L2(_data.getNumDimensions(), _data.getNumPoints(), _dataNormed.data());

            _faissData = utils::DataView(&_dataNormed, _data.numPoints, _data.numDimensions);
        }
        else {
            _faissData = utils::DataView(_data.dataVec, _data.numPoints, _data.numDimensions);
        }

    }

    void NearestNeighbors::squareRootOfAllDistances()
    {
        // take square root of squared euclidean distances
        SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(_knnGraph.knnDistances.size()); i++) {
            _knnGraph.knnDistances[i] = std::sqrt(_knnGraph.knnDistances[i]);
        }
    }

    void NearestNeighbors::ensureFloatingPointIntegrity()
    {
        constexpr float epsilon = std::numeric_limits<float>::epsilon();

        SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(_knnGraph.knnDistances.size()); i++) {
            if (_knnGraph.knnDistances[i] <= epsilon)
                _knnGraph.knnDistances[i] = 0.f;
        }
    }

    void NearestNeighbors::checkDistancesAreNonDecreasing(utils::Graph& g)
    {
        Log::info("NearestNeighbors::checkDistancesAreNonDecreasing");

        auto is_non_decreasing = [](std::span<const float> values) {
            return std::adjacent_find(SPH_PARALLEL_EXECUTION
                values.begin(), values.end(), std::greater<float>()) == values.end();
            };

        for (int64_t i = 0; i < g.getNumPoints(); i++) {
            if (!is_non_decreasing(g.getDistances(i))) {

                Log::warn("Before");
                utils::print(g.getDistances(i));
                utils::print(g.getNeighbors(i));

                utils::synchronizedSort(g.getDistancesRef(i), g.getNeighborsRef(i));

                if (!is_non_decreasing(g.getDistances(i))){
                    Log::warn("Before");
                    utils::print(g.getDistances(i));
                    utils::print(g.getNeighbors(i));
                }
            }
        }
    }

    bool NearestNeighbors::checkAllNeighborsExist()
    {
        Log::info("NearestNeighbors::checkAllNeighborsExist");

        const auto missingPoints = utils::checkAllNeighborsExist(_knnGraph.getGraphView());

        if (missingPoints.has_value())
            return false;

        const auto& nns = _knnGraph.getNns();

        for (int64_t p = 0; p < _knnGraph.getNumPoints(); ++p)
        {
            if (nns[p] != static_cast<int64_t>(_nns.numNearestNeighbors)) {
                return false;
            }
        }

        return true;
    }

    std::filesystem::path NearestNeighbors::getConnectedComponentsCacheFile(bool symmetric) const
    {
        std::string ccFolder = "nbSym" + std::to_string(static_cast<uint32_t>(symmetric));
        auto ccPath = std::filesystem::path(_ccCache.path) / ccFolder;

        return std::filesystem::path(_ccCache.path);
    }

    bool NearestNeighbors::checkCCCachePath(const std::filesystem::path& p) const
    {
        bool res = false;

        if (!_ccCache.cacheActive)
            return res;

        std::string fileNameBase = std::filesystem::path(_ccCache.fileName).stem().string();
        if (!utils::stringContains(_ccCache.path, fileNameBase))
            return res;

        res = std::filesystem::exists(p);
        if (!res)
            res = std::filesystem::create_directories(p);

        return res;
    }

    std::pair<int64_t, std::shared_ptr<std::vector<int64_t>>> NearestNeighbors::computeConnectedComponents()
    {
        Log::info("NearestNeighbors::computeConnectedComponents: started");

        if (!_hasNeighbors)
            Log::warn("NearestNeighbors::computeConnectedComponents: _knn graph is not computed yet, call compute() fist");

        if (_knnGraph.knnIndices.empty())
            Log::warn("NearestNeighbors::computeConnectedComponents: _knn graph does not have any points");

        utils::GraphView dataGraph{};

        if (_hasSymmetric)
        {
            Log::info("NearestNeighbors::computeConnectedComponents: using symmetric knn graph");
            dataGraph = _symGraph.getGraphView();
        }
        else
        {
            Log::info("NearestNeighbors::computeConnectedComponents: using knn graph");
            dataGraph = _knnGraph.getGraphView();
        }

        assert(dataGraph.isValid());

        const auto cs_cc_path = getConnectedComponentsCacheFile(_hasSymmetric);
        const auto cs_cc_file = cs_cc_path / "cc.cache";
        const auto cs_cc_nums_file = cs_cc_path / "cc_num.cache";

        auto loadCC = [this, &cs_cc_path, &cs_cc_file, &cs_cc_nums_file]() -> bool {
            if (!checkCCCachePath(cs_cc_path))
                return false;

            if (!std::filesystem::exists(cs_cc_file))
                return false;

            if (!std::filesystem::exists(cs_cc_nums_file))
                return false;

            Log::info("NearestNeighbors::computeConnectedComponents: loading {0}", cs_cc_file.string());
            auto success_cc = utils::loadCompressedVecFromBinary(cs_cc_file.string(), *_connectedComponents);

            Log::info("NearestNeighbors::computeConnectedComponents: loading {0}", cs_cc_nums_file.string());
            vi64 numsCC;
            auto success_cc_nums = utils::loadCompressedVecFromBinary(cs_cc_nums_file.string(), numsCC);
            assert(numsCC.size() == 1);
            _numConnectedComponents = numsCC[0];

            if (success_cc && success_cc_nums)
            {
                Log::info("NearestNeighbors::computeConnectedComponents: successfully loaded CC");
                return true;
            }

            return false;
            };

        auto saveCC = [this, cs_cc_path, &cs_cc_file, &cs_cc_nums_file]() {
            if (!_ccCache.cacheActive)
                return;

            if (!std::filesystem::exists(cs_cc_path))
                Log::info("NearestNeighbors::computeConnectedComponents: cannot save CC since save path does not exist");

            Log::info("NearestNeighbors::computeConnectedComponents: writing {0}", cs_cc_file.string());
            bool success_cc = utils::writeCompressedVecToBinary(cs_cc_file.string(), *_connectedComponents);

            Log::info("NearestNeighbors::computeConnectedComponents: writing {0}", cs_cc_nums_file.string());
            bool success_cc_nums = utils::writeCompressedVecToBinary(cs_cc_nums_file.string(), vi64{ _numConnectedComponents });

            if (success_cc && success_cc_nums)
                Log::info("NearestNeighbors::computeConnectedComponents: successfully saved CC");

            };

        auto computeCC = [this, saveCC, &dataGraph]() {
            if (_hasSymmetric)
                _numConnectedComponents = utils::labelGraphStrongComponents(dataGraph, *_connectedComponents);
            else
                _numConnectedComponents = utils::labelGraphWeakComponents(dataGraph, *_connectedComponents);

            saveCC();
            };

        if (!loadCC())
            computeCC();

        Log::info("NearestNeighbors::computeConnectedComponents: {0} connected components", _numConnectedComponents);
        assert(_connectedComponents->size() == static_cast<size_t>(_data.getNumPoints()));

        return std::make_pair(_numConnectedComponents, _connectedComponents);
    }

    const utils::GraphView NearestNeighbors::computeSymmetrizedNnGraph()
    {
        Log::info("NearestNeighbors::computeSymmetrizedNnGraph");

        if (!_hasNeighbors)
        {
            Log::warn("NearestNeighbors::computeSymmetrizedNnGraph: _knn graph is not computed yet, call compute() fist, doing nothing");
            return {};
        }

        if (_knnGraph.getNumPoints() == 0)
        {
            Log::warn("NearestNeighbors::computeSymmetrizedNnGraph: _knn graph does not have any points, doing nothing");
            return {};
        }

        assert(_knnGraph.isValid());

        _symGraph.numPoints = _knnGraph.getNumPoints();

        auto saveSymmetricGraph = [this]() {
            if (!_ccCache.cacheActive)
                return;

            if (!std::filesystem::exists(fullCachePath()))
                Log::info("NearestNeighbors::computeSymmetrizedNnGraph: cannot save connected graph since save path does not exist");

            const auto fileName_graph = pathSymKnn();

            Log::info("NearestNeighbors::computeSymmetrizedNnGraph: writing {0}", fileName_graph);
            auto symmetricGraphView = _symGraph.getGraphView();
            bool success_cache_sym_graph = writeCompressedGraphToBinary(fileName_graph, symmetricGraphView);

            if (success_cache_sym_graph)
                Log::info("NearestNeighbors::computeSymmetrizedNnGraph: successfully saved connected graph");
            else
                Log::warn("NearestNeighbors::computeSymmetrizedNnGraph: could not save connected graph");

            };

        auto loadSymmetricGraph = [this]() -> bool {
            if (!std::filesystem::exists(fullCachePath()))
                return false;

            const auto fileName_graph = pathSymKnn();

            Log::info("NearestNeighbors::computeSymmetrizedNnGraph: loading {0}", fileName_graph);
            bool success_cache_sym_graph = loadCompressedGraphFromBinary(fileName_graph, _symGraph);

            if (!success_cache_sym_graph)
            {
                Log::info("NearestNeighbors::computeSymmetrizedNnGraph: could not load symmetric graph");
                return false;
            }

            Log::info("NearestNeighbors::computeSymmetrizedNnGraph: loaded symmetric graph");

            return true;
            };

        if (!loadSymmetricGraph())
        {
            Log::info("NearestNeighbors::computeSymmetrizedNnGraph: computing...");
            _symGraph = utils::symmetrizeGraph(_knnGraph.getGraphView());
            saveSymmetricGraph();
        }

        checkDistancesAreNonDecreasing(_symGraph);
        auto numAdjustedPoints = utils::ensureClosestPointIsSelf(&_symGraph);

        if (numAdjustedPoints > 0)
            Log::info("NearestNeighbors::ensureClosestPointIsSelf: {} points were adjusted (of {})", numAdjustedPoints, _knnGraph.getNumPoints());

        _hasSymmetric = true;

        assert(_symGraph.isValid());
        assert(_symGraph.hasUniqueNeighbors());

        printSymGraphInfo("computeSymmetrizedNnGraph:");

        return _symGraph.getGraphView();
    }

    const utils::GraphView NearestNeighbors::connectComponents()
    {
        Log::info("NearestNeighbors::connectComponents");

        if (!hasComponentsComputed())
        {
            Log::info("NearestNeighbors::connectComponents: No connected components present yet, computing them first...");
            computeConnectedComponents();
        }

        utils::GraphView dataGraph{};

        if (_hasSymmetric)
        {
            Log::info("NearestNeighbors::connectComponents: using symmetric knn graph");
            dataGraph = _symGraph.getGraphView();
            _connectedGraph.nns.assign(_symGraph.getNns().begin(), _symGraph.getNns().end());
        }
        else
        {
            Log::info("NearestNeighbors::connectComponents: using knn graph");
            dataGraph = _knnGraph.getGraphView();
            _connectedGraph.nns.assign(_knnGraph.getNumPoints(), _knnGraph.getK());
        }

        // Setup: Copy knn graph to general graph
        _connectedGraph.knnIndices.assign(dataGraph.getKnnIndices().cbegin(), dataGraph.getKnnIndices().cend());
        _connectedGraph.knnDistances.assign(dataGraph.getKnnDistances().cbegin(), dataGraph.getKnnDistances().cend());
        _connectedGraph.numPoints = dataGraph.getNumPoints();
        _connectedGraph.symmetric = dataGraph.isSymmetric();
        _connectedGraph.updateOffsets();

        if (_numConnectedComponents == 1)
        {
            Log::info("NearestNeighbors::connectComponents: All points are connected, no need to add connections since _numConnectedComponents = {}", _numConnectedComponents);

            _hasConnected = true;
            printConGraphInfo("connectComponents:");

            return _connectedGraph.getGraphView();
        }

        assert(_connectedComponents->size() == static_cast<size_t>(_faissData.getNumPoints()));
        assert(_data.getNumPoints() == _faissData.getNumPoints());
        assert(_data.getNumPoints() == dataGraph.getNumPoints());

        const auto d = static_cast<int>(_faissData.getNumDimensions());
        const auto m = static_cast<faiss::MetricType>(_faissMetric);
        const auto cs_spgTree_path = getConnectedComponentsCacheFile(_hasSymmetric);
        const std::string cs_spgTree_name = "spgTree.cache";
        const std::string cs_cgraph_name = "connected_graph";

        // add edge to graph
        auto insertDistance = [](utils::Graph& g, int64_t a, int64_t b, float d) {
            if (a == b)
                return;

            auto neighborsA = g.getNeighborsRef(a);
            if (utils::contains(neighborsA, b))
                return;

            auto distancesA = g.getDistancesRef(a);
            auto it = std::upper_bound(distancesA.begin(), distancesA.end(), d);
            auto index = std::distance(distancesA.begin(), it);

            if (index == 0) // Do not insert in first position
                index++;    // Since first neighbor should remain the point itself

            const auto skip = g.offsets[a];
            assert(skip == std::accumulate(g.nns.begin(), g.nns.begin() + a, 0ll));

            g.knnDistances.insert(g.knnDistances.begin() + skip + index, d);
            g.knnIndices.insert(g.knnIndices.begin() + skip + index, b);
            g.nns[a]++;
            g.updateOffsets();

            assert(g.isValid());
            };

        // returns all values and global Ids for a CC
        auto getComponentValues = [this](int64_t component) -> std::pair<std::vector<float>, std::vector<int64_t>> {
            std::vector<float> compValues;
            std::vector<int64_t> compIDs;

            for (int64_t id : std::views::iota(0ll, _faissData.getNumPoints()))
            {
                if (component != (*_connectedComponents)[id])
                    continue;

                auto values = _faissData.getValuesAt(id);
                compValues.insert(compValues.end(), values.begin(), values.end());
                compIDs.push_back(id);
            }

            return { std::move(compValues), std::move(compIDs) };
            };

        // finds the smallest distance between any two points from the two given components and introduces a bidirectional connection
        auto insertConnectionsBetweenComponents = [this, &d, &m, getComponentValues, insertDistance](int64_t compA, int64_t compB) {
            // retrieve component values
            auto [valsA, idsA] = getComponentValues(compA);
            auto [valsB, idsB] = getComponentValues(compB);

            // train an index on the points in A
            int nlist = 100;
            int nprobe = 10;
            int64_t npA = valsA.size() / d;
            int64_t npB = valsB.size() / d;
            vi64    comp_indices(npB);
            vf32    comp_distances(npB);

            faiss::ClusteringParameters cl; // helper struct, only used to access value of min_points_per_centroid

            // For each point in B find the (1) closest neighbor in A
            if (npA <= static_cast<int64_t>(nlist) * nprobe || npA <= static_cast<int64_t>(nlist) * cl.min_points_per_centroid)
            {
                auto iFlat = faiss::IndexFlat(d, m);
                iFlat.add(npA, valsA.data());
                iFlat.search(
                    npB,
                    valsB.data(),
                    1,
                    comp_distances.data(),
                    comp_indices.data()
                );
            }
            else
            {
                auto iFlat = faiss::IndexFlat(d, m);
                auto iIVFFlat = faiss::IndexIVFFlat(&iFlat, d, nlist, m);
                iIVFFlat.nprobe = nprobe;
                iIVFFlat.train(npA, valsA.data());
                iIVFFlat.add(npA, valsA.data());

                iIVFFlat.search(
                    npB,
                    valsB.data(),
                    1,
                    comp_distances.data(),
                    comp_indices.data()
                );
            }

            // Find the smallest distance and the corresponding pair of points
            float minDistance = std::numeric_limits<float>::max();
            int64_t minIdA = -1;
            int64_t minIdB = -1;

            for (int64_t i = 0; i < npB; ++i) {
                if (comp_distances[i] < minDistance) {
                    minDistance = comp_distances[i];
                    minIdA = comp_indices[i];
                    minIdB = i;
                }
            }

            // Add distance to general graph, both directions
            int64_t globIdA = idsA[minIdA];
            int64_t globIdB = idsB[minIdB];

            insertDistance(_connectedGraph, globIdA, globIdB, minDistance);
            insertDistance(_connectedGraph, globIdB, globIdA, minDistance);
            };

        // compute center of each connected component
        auto computeCentersOfMass = [this, getComponentValues](int dim) -> vf32 {
            Log::info("NearestNeighbors::connectComponents: compute center of each connected component");

            vf32 centers(_numConnectedComponents * dim);
            utils::ProgressBar progress(_numConnectedComponents);

            SPH_PARALLEL
            for (int64_t cc = 0; cc < _numConnectedComponents; cc++)
            {
                const auto [vals, ids] = getComponentValues(cc);
                assert(vals.size() == ids.size() * dim);

                const auto mean = utils::computeMean(vals, static_cast<size_t>(dim));
                assert(mean.size() == static_cast<size_t>(dim));

                std::span<float> destSpan{ centers.data() + cc * dim, centers.data() + cc * dim + dim };
                std::copy(mean.begin(), mean.end(), destSpan.begin());

                progress.update();
            }
            progress.finish();
            return centers;
            };

        // edges between components
        auto computeSpanningTree = [this, computeCentersOfMass](int dim) -> std::vector<utils::BoostEdgeFullU> {
            // heuristic: use center of mass of each component
            const vf32 centers = computeCentersOfMass(dim);

            // create fully connected graph between center of masses
            Log::info("NearestNeighbors::connectComponents: create fully-connected boost helper graph between components");
            utils::BoostGraphFullU bgraph(_numConnectedComponents);
            utils::ProgressBar progress(_numConnectedComponents);
            for (int64_t i = 0; i < _numConnectedComponents; ++i) {
                for (int64_t j = i + 1; j < _numConnectedComponents; ++j) {
                    boost::add_edge(i, j, utils::L2(centers.data() + i * dim, centers.data() + j * dim, dim), bgraph);
                }
                progress.update();
            }
            progress.finish();

            assert(boost::num_edges(bgraph) == static_cast<size_t>(_numConnectedComponents * (_numConnectedComponents - 1) / 2));

            // compute minimum spanning tree and add those edges
            Log::info("NearestNeighbors::connectComponents: compute minimum spanning tree between components");
            std::vector<utils::BoostEdgeFullU> spgTree;
            boost::kruskal_minimum_spanning_tree(bgraph, std::back_inserter(spgTree));

            return spgTree;
            };

        auto saveSpanningTree = [this, &cs_spgTree_path, &cs_spgTree_name](const std::vector<utils::BoostEdgeFullU>& spgTree) {
            if (!_ccCache.cacheActive)
                return;

            if (!std::filesystem::exists(cs_spgTree_path))
                Log::info("NearestNeighbors::connectComponents: cannot save spanning tree since save path does not exist");

            // convert to vec of int
            vi64 spgTreeIdx;
            for (const auto& edge : spgTree)
            {
                spgTreeIdx.push_back(edge.m_source);
                spgTreeIdx.push_back(edge.m_target);
            }

            // save to disk
            const auto fileName = cs_spgTree_path / cs_spgTree_name;
            Log::info("NearestNeighbors::connectComponents: writing {0}", fileName.string());
            bool success_spgTree = utils::writeCompressedVecToBinary(fileName.string(), spgTreeIdx);

            if (success_spgTree)
                Log::info("NearestNeighbors::connectComponents: successfully saved spanning tree");
            else
                Log::warn("NearestNeighbors::connectComponents: could not save spanning tree");
            };

        auto loadSpanningTree = [this, &cs_spgTree_path, &cs_spgTree_name](std::vector<utils::BoostEdgeFullU>& spgTree) -> bool {
            if (!checkCCCachePath(cs_spgTree_path))
                return false;

            const auto fileName = cs_spgTree_path / cs_spgTree_name;
            if (!std::filesystem::exists(fileName))
                return false;

            // load from disk
            Log::info("NearestNeighbors::connectComponents: loading {0}", fileName.string());
            vi64 spgTreeIdx;
            auto success_spgTree = utils::loadCompressedVecFromBinary(fileName.string(), spgTreeIdx);

            if (!success_spgTree)
            {
                Log::info("NearestNeighbors::connectComponents: could not load spanning tree");
                return false;
            }

            // convert vec of int to vec of boost edges
            assert(spgTreeIdx.size() % 2 == 0);
            spgTree.clear();
            for (size_t i = 0; i < spgTreeIdx.size(); i += 2)
                spgTree.push_back(utils::BoostEdgeFullU(true, spgTreeIdx[i], spgTreeIdx[i + 1]));

            Log::info("NearestNeighbors::connectComponents: loaded spanning tree");

            return true;
            };

        auto computeConnectedGraph = [this, computeSpanningTree, insertConnectionsBetweenComponents, loadSpanningTree, saveSpanningTree, d]() {
            std::vector<utils::BoostEdgeFullU> spanningTree;

            if (!loadSpanningTree(spanningTree))
            {
                spanningTree = computeSpanningTree(d);
                saveSpanningTree(spanningTree);
            }

            assert(spanningTree.size() == static_cast<size_t>(_numConnectedComponents - 1));

            Log::info("NearestNeighbors::connectComponents: add minimum spanning tree edges (between components) to data graph");
            utils::ProgressBar progress(spanningTree.size());
            for (const auto& newEdge : spanningTree)
            {
                insertConnectionsBetweenComponents(newEdge.m_source, newEdge.m_target);
                progress.update();
            }
            progress.finish();
            };

        auto saveConnectedGraph = [this, &cs_spgTree_path, &cs_cgraph_name]() {
            if (!_ccCache.cacheActive)
                return;

            if (!std::filesystem::exists(cs_spgTree_path))
                Log::info("NearestNeighbors::connectComponents: cannot save connected graph since save path does not exist");

            const auto fileName_graph = cs_spgTree_path / cs_cgraph_name;

            Log::info("NearestNeighbors::connectComponents: writing {0}", fileName_graph.string());
            auto connectedGraphView = _connectedGraph.getGraphView();
            bool success_cs_cgraph = writeCompressedGraphToBinary(fileName_graph.string(), connectedGraphView);

            if (success_cs_cgraph)
                Log::info("NearestNeighbors::connectComponents: successfully saved connected graph");
            else
                Log::warn("NearestNeighbors::connectComponents: could not save connected graph");

            };

        auto loadConnectedGraph = [this, &cs_spgTree_path, &cs_cgraph_name]() -> bool {
            if (!checkCCCachePath(cs_spgTree_path))
                return false;

            const auto fileName_graph = cs_spgTree_path / cs_cgraph_name;

            Log::info("NearestNeighbors::connectComponents: loading {0}", fileName_graph.string());
            bool success_cs_cgraph = loadCompressedGraphFromBinary(fileName_graph.string(), _connectedGraph);

            if (!success_cs_cgraph)
            {
                Log::info("NearestNeighbors::connectComponents: could not load connected graph");
                return false;
            }

            Log::info("NearestNeighbors::connectComponents: loaded connected graph");

            return true;
            };

        Log::info("NearestNeighbors::connectComponents: old number of connections is {}", _connectedGraph.getNumEdges());
        if (!loadConnectedGraph())
        {
            computeConnectedGraph();
            saveConnectedGraph();
        }
        Log::info("NearestNeighbors::connectComponents: new number of connections is {}", _connectedGraph.getNumEdges());

        checkDistancesAreNonDecreasing(_connectedGraph);
        auto numAdjustedPoints = utils::ensureClosestPointIsSelf(&_connectedGraph);

        if (numAdjustedPoints > 0)
            Log::info("NearestNeighbors::ensureClosestPointIsSelf: {} points were adjusted (of {})", numAdjustedPoints, _knnGraph.getNumPoints());

#if SPH_DEBUG
        {
            Log::debug("NearestNeighbors::connectComponents: checking that the connected graph has indeed only one connected component...");

            assert(_connectedGraph.isValid());
            assert(_connectedGraph.hasUniqueNeighbors());
            int64_t numConnectedComponents = 0; 
            if (_hasSymmetric)
                numConnectedComponents = utils::labelGraphStrongComponents(_connectedGraph.getGraphView());
            else
                numConnectedComponents = utils::labelGraphWeakComponents(_connectedGraph.getGraphView());

            assert(numConnectedComponents == 1);
        }
#endif

        _hasConnected = true;
        printConGraphInfo("connectComponents:");

        return _connectedGraph.getGraphView();
    }

    /// /////// ///
    /// Cashing ///
    /// /////// ///

    bool NearestNeighbors::checkCacheParameters(const std::string& fileName) const {
        auto [parameters, successLoadingParameters] = utils::loadJsonFromDisk(fileName);

        if (!successLoadingParameters)
            return false;

        if (!isVersionCompatible(parameters))
            return false;

        if (!utils::checkEntry("Input data name", parameters, _cacheFileName)) return false;

        if (!utils::checkSettings(parameters, _nns)) return false;

        Log::info("Parameters of cache correspond to current settings.");

        return true;
    }

    bool NearestNeighbors::writeCacheParameters(const std::string& fileName) const
    {
        Log::info("NearestNeighbors::writeCacheParameters: write to " + fileName);

        // store parameters in json file
        nlohmann::json parameters;
        parameters["## VERSION ##"] = _cacheParameterVersion;

        parameters["Input data name"] = _cacheFileName;

        utils::addToJson(_nns, parameters);

        // Write to file
        bool successWritingParameters = utils::writeJsonToDisk(fileName, parameters);

        if (!successWritingParameters)
            return false;

        return true;
    }

    bool NearestNeighbors::loadCache()
    {
        if (!_cacheIsActive)
        {
            Log::info("NearestNeighbors::loadCache: Caching is not active. Use setCachingActive(true) if desired.");
            return cachingFailure();
        }

        if (!cacheDependencyIsValid())
        {
            Log::info("ImageHierarchy::loadCache: Dependency not loaded from cache.");
            return cachingFailure();
        }

        Log::info("NearestNeighbors::loadCache: attempt to load cache from " + _cachePath.string());

        if (!std::filesystem::exists(_cachePath))
        {
            Log::warn("Loading cache failed: does not exist.");
            return cachingFailure();
        }

        const auto pParameter = pathParameter();
        const auto pDataKnn   = pathDataKnn();
        const auto pSymKnn    = pathSymKnn();
        const auto pConKnn    = pathConKnn();
        const auto pNumComp   = pathNumComp();
        const auto pConComp   = pathConComp();

        if (!checkCacheParameters(pParameter))
        {
            Log::warn("Loading cache failed: Current settings are different from cached parameters.");
            return cachingFailure();
        }

        if (!utils::loadCompressedGraphFromBinary(pDataKnn, _knnGraph))
        {
            Log::warn("Loading cache failed: " + pDataKnn);
            return cachingFailure();
        }

        if (!utils::loadCompressedGraphFromBinary(pSymKnn, _symGraph))
        {
            Log::warn("Loading cache failed: " + pSymKnn);
            return cachingFailure();
        }

        if (!utils::loadCompressedGraphFromBinary(pConKnn, _connectedGraph))
        {
            Log::warn("Loading cache failed: " + pConKnn);
            return cachingFailure();
        }

        std::vector<int64_t> numConnectedComponents;
        if (!utils::loadVecFromBinary(pNumComp, numConnectedComponents))
        {
            Log::warn("Loading cache failed: " + pNumComp);
            return cachingFailure();
        }
        _numConnectedComponents = numConnectedComponents[0];

        if (!utils::loadCompressedVecFromBinary(pConComp, *_connectedComponents.get()))
        {
            Log::warn("Loading cache failed: " + pConComp);
            return cachingFailure();
        }

        initFaissData();

        _hasNeighbors = true;
        printKnnGraphInfo("loadCache:");

        if (_symGraph.getNumPoints() == _knnGraph.getNumPoints())
        {
            _hasSymmetric = true;
            printSymGraphInfo("loadCache:");
        }

        if(_connectedGraph.getNumPoints() == _knnGraph.getNumPoints())
        {
            _hasConnected = true;
            printConGraphInfo("loadCache:");
        }

        Log::info("NearestNeighbors::loadCache: finished");
        return cachingSuccess();
    }

    bool NearestNeighbors::writeCache() const
    {
        if (!mayCache("NearestNeighbors"))
            return false;

        Log::info("NearestNeighbors::writeCache: save cache to " + _cachePath.string());

        const auto pParameter = pathParameter();
        const auto pDataKnn   = pathDataKnn();
        const auto pSymKnn    = pathSymKnn();
        const auto pConKnn    = pathConKnn();
        const auto pNumComp   = pathNumComp();
        const auto pConComp   = pathConComp();

        const bool successParam = writeCacheParameters(pParameter);

        const auto knnGraphView = _knnGraph.getGraphView();
        const bool successKnn   = utils::writeCompressedGraphToBinary(pDataKnn, knnGraphView);

        const auto symGraphView = _symGraph.getGraphView();
        const bool successSym   = utils::writeCompressedGraphToBinary(pSymKnn, symGraphView);

        const auto conGraphView = _connectedGraph.getGraphView();
        const bool successCon   = utils::writeCompressedGraphToBinary(pConKnn, conGraphView);

        std::vector<int64_t> numConnectedComponents = { _numConnectedComponents };
        bool successNumComp = utils::writeVecToBinary(pNumComp, numConnectedComponents);

        const vi64* conComponents = _connectedComponents.get();
        bool successConComp = utils::writeCompressedVecToBinary(pConComp, *conComponents);

        Log::info("NearestNeighbors::writeCache: finished");
        return successParam && successKnn && successSym && successCon && successNumComp && successConComp;
    }

} // namespace sph
