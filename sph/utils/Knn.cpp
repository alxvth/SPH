#include "Knn.hpp"

#include "Logger.hpp"
#include "Timer.hpp"

#include <cmath>

#pragma warning(disable:4244)       // faiss internal: conversion from 'faiss::idx_t' to 'int', possible loss of data
#pragma warning(disable:4251)       // faiss internal: std class ... needs to have dll-interface to be used by clients of struct 'faiss::InterruptCallback'
#pragma warning(disable:4068)       // MSVC: unknown pragma
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter" // faiss internal: unused parameters id, nbits, nbits_in
#include <faiss/utils/Heap.h>       // faiss::float_maxheap_array_t
#include <faiss/utils/distances.h>  // faiss::knn_L2sqr
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/Clustering.h>
#include <faiss/MetricType.h>
#pragma GCC diagnostic pop
#pragma warning(default:4068)
#pragma warning(default:4251)
#pragma warning(default:4244)

/*
Some resources:
- https://github.com/facebookresearch/faiss/wiki/Faster-search
- https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- https://github.com/matsui528/faiss_tips
*/

namespace faiss {
    std::string asText(const MetricType& obj) {
        switch (obj)
        {
        case faiss::MetricType::METRIC_INNER_PRODUCT:
            return "METRIC_INNER_PRODUCT";
        case faiss::MetricType::METRIC_L2:
            return "METRIC_L2";
        case faiss::MetricType::METRIC_L1:
            return "METRIC_L1";
        default:
            return std::to_string(obj);
        }

        return std::to_string(obj);
    }
} // faiss

namespace sph::utils {

    using CMaxf = faiss::CMax<float, int64_t>;
    using CMinf = faiss::CMin<float, int64_t>;

    void computeButeForce(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph)
    {
        utils::ScopedTimer<std::chrono::seconds> myTimer("Knn brute force (exact)", "sec");

        const auto m = static_cast<faiss::MetricType>(faissMetric);

        Log::info("computeButeForce with k = {0} and metric {1}", nn, faiss::asText(m));

        switch (m)
        {
        case faiss::MetricType::METRIC_L2:
        {
            faiss::HeapArray<CMaxf> heapArrMax = {};            // create new index
            heapArrMax.nh = data.getNumPoints();                // number of heaps
            heapArrMax.k = nn;                                  // allocated size per heap
            heapArrMax.ids = knnGraph.knnIndices.data();        // identifiers (size nh * k)
            heapArrMax.val = knnGraph.knnDistances.data();      // values (distances or similarities), size nh * k

            faiss::knn_L2sqr(
                data.data(),      // query vectors
                data.data(),      // database vectors
                data.getNumDimensions(),
                data.getNumPoints(),
                data.getNumPoints(),
                &heapArrMax
            );

            break;
        }
        case faiss::MetricType::METRIC_INNER_PRODUCT:
        {
            // ensure that the data is normalized for cosine
            faiss::HeapArray<CMinf> heapArrMin = {};            // create new index
            heapArrMin.nh = data.getNumPoints();                // number of heaps
            heapArrMin.k = nn;                                  // allocated size per heap
            heapArrMin.ids = knnGraph.knnIndices.data();        // identifiers (size nh * k)
            heapArrMin.val = knnGraph.knnDistances.data();      // values (distances or similarities), size nh * k

            faiss::knn_inner_product(
                data.data(),      // query vectors
                data.data(),      // database vectors
                data.getNumDimensions(),
                data.getNumPoints(),
                data.getNumPoints(),
                &heapArrMin);
            
            break;
        }
        default:
            Log::warn("computeButeForce: not implemented: {}", faiss::asText(m));
        } // switch

    }

    void computeIndexFlat(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph)
    {
        utils::ScopedTimer<std::chrono::seconds> myTimer("Knn IndexFlat (exhaustive)", "sec");

        const auto m = static_cast<faiss::MetricType>(faissMetric);

        Log::info("computeIndexFlat: k = {0} and metric {1}", nn, faiss::asText(m));

        auto d = static_cast<int>(data.getNumDimensions());
        auto np = data.getNumPoints();
        auto xb = data.data();

        auto iFlat = faiss::IndexFlat(d, m);

        Log::info("computeIndexFlat: adding...");
        iFlat.add(np, xb);

        Log::info("computeIndexFlat: searching...");
        iFlat.search(
            np,
            xb,
            nn,
            knnGraph.knnDistances.data(),
            knnGraph.knnIndices.data()
        );

    }

    void computeIndexIVFFlat(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph)
    {
        utils::ScopedTimer<std::chrono::seconds> myTimer("Knn IndexIVFFlat (approximate)", "sec");

        const auto m = static_cast<faiss::MetricType>(faissMetric);

        Log::info("computeIndexIVFFlat: k = {0} and metric {1}", nn, faiss::asText(m));

        int nlist = static_cast<int>(std::max(100.0, std::sqrt(data.getNumPoints()))); // number of cells for space partitioning
        int nprobe = static_cast<int>(std::sqrt(nlist));                                      // number of cells (out of nlist) that are visited to perform a search

        auto d = static_cast<int>(data.getNumDimensions());
        auto np = data.getNumPoints();
        auto xb = data.data();

        auto iFlat = faiss::IndexFlat(d, m);
        auto iIVFFlat = faiss::IndexIVFFlat(&iFlat, d, nlist, m);   // create new index
        assert(!iIVFFlat.is_trained);
        iIVFFlat.nprobe = nprobe;

        assert(!iIVFFlat.is_trained);

        Log::info("computeIndexIVFFlat: training...");
        iIVFFlat.train(np, xb);
        assert(iIVFFlat.is_trained);

        Log::info("computeIndexIVFFlat: adding...");
        iIVFFlat.add(np, xb);

        Log::info("computeIndexIVFFlat: searching...");
        iIVFFlat.search(
            np,
            xb,
            nn,
            knnGraph.knnDistances.data(),
            knnGraph.knnIndices.data()
        );
    }

    void computeIndexHNSW(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph)
    {
        utils::ScopedTimer<std::chrono::seconds> myTimer("Knn IndexHNSWFlat (approximate)", "sec");

        const auto m = static_cast<faiss::MetricType>(faissMetric);

        Log::info("computeIndexHNSW: k = {0} and metric {1}", nn, faiss::asText(m));

        int M = 64;                 // number of neighbors used in the graph. Larger is more accurate but uses more memory
        int efConstruction = 64;    // depth of exploration at add time
        int efSearch = 128;         // depth of exploration of the search

        auto d = static_cast<int>(data.getNumDimensions());
        auto np = data.getNumPoints();
        auto xb = data.data();

        auto distances = knnGraph.knnDistances.data();
        auto labels = knnGraph.knnIndices.data();

        auto iHNSWFlat = faiss::IndexHNSWFlat(d, M, m);
        iHNSWFlat.hnsw.efConstruction = efConstruction;
        iHNSWFlat.hnsw.efSearch = efSearch;

        // HNSW does not require training

        Log::info("computeIndexHNSW: adding...");
        iHNSWFlat.add(np, xb);

        Log::info("computeIndexHNSW: searching...");
        iHNSWFlat.search(
            np,
            xb,
            nn,
            distances,
            labels
        );

        const auto missingPoints = checkAllNeighborsExist(knnGraph.getGraphView());

        if (missingPoints.has_value())
        {
            Log::info("computeIndexHNSW: try to fill missing neighbors for {} points", missingPoints.value().size());

            iHNSWFlat.hnsw.efSearch = iHNSWFlat.hnsw.efSearch * 4;

            for (const int64_t p : missingPoints.value()) {
                xb = data.data() + p * data.getNumDimensions();
                distances = knnGraph.knnDistances.data() + p * nn;
                labels = knnGraph.knnIndices.data() + p * nn;
                iHNSWFlat.search(
                    1,
                    xb,
                    nn,
                    distances,
                    labels
                );
            }

            const auto missingPoints2 = checkAllNeighborsExist(knnGraph.getGraphView());

            if (missingPoints2.has_value())
                Log::warn("computeIndexHNSW: failed to fill missing neighbors...");
            else
                Log::info("computeIndexHNSW: succeeded to fill missing neighbors!");

        }

    }

    void computeIndexHNSWSQ(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph)
    {
        utils::ScopedTimer<std::chrono::seconds> myTimer("Knn IndexHNSWSQ (approximate)", "sec");

        const auto m = static_cast<faiss::MetricType>(faissMetric);

        Log::info("computeIndexHNSWSQ: k = {0} and metric {1}", nn, faiss::asText(m));

        int M = 64;                 // number of neighbors used in the graph. Larger is more accurate but uses more memory
        int efConstruction = 64;    // depth of exploration at add time
        int efSearch = 128;         // depth of exploration of the search
        auto scalarQuant = faiss::ScalarQuantizer::QuantizerType::QT_8bit;

        auto d = static_cast<int>(data.getNumDimensions());
        auto np = data.getNumPoints();
        auto xb = data.data();

        auto distances = knnGraph.knnDistances.data();
        auto labels = knnGraph.knnIndices.data();

        auto iHNSWSQ = faiss::IndexHNSWSQ(d, scalarQuant, M, m);
        iHNSWSQ.hnsw.efConstruction = efConstruction;
        iHNSWSQ.hnsw.efSearch = efSearch;

        assert(!iHNSWSQ.is_trained);

        Log::info("computeIndexHNSWSQ: training...");
        iHNSWSQ.train(np, xb);

        assert(iHNSWSQ.is_trained);

        Log::info("computeIndexHNSWSQ: adding...");
        iHNSWSQ.add(np, xb);

        Log::info("computeIndexHNSWSQ: searching...");
        iHNSWSQ.search(
            np,
            xb,
            nn,
            distances,
            labels
        );

        const auto missingPoints = checkAllNeighborsExist(knnGraph.getGraphView());

        if (missingPoints.has_value())
        {
            Log::info("computeIndexHNSWSQ: try to fill missing neighbors for {} points...", missingPoints.value().size());

            iHNSWSQ.hnsw.efSearch = iHNSWSQ.hnsw.efSearch * 4;

            for (const int64_t p : missingPoints.value()) {
                xb = data.data() + p * data.getNumDimensions();
                distances = knnGraph.knnDistances.data() + p * nn;
                labels = knnGraph.knnIndices.data() + p * nn;
                iHNSWSQ.search(
                    1,
                    xb,
                    nn,
                    distances,
                    labels
                );
            }

            const auto missingPoints2 = checkAllNeighborsExist(knnGraph.getGraphView());

            if(missingPoints2.has_value())
                Log::warn("computeIndexHNSWSQ: failed to fill missing neighbors...");
            else
                Log::info("computeIndexHNSWSQ: succeeded to fill missing neighbors!");

        }

    }


    void computeIndexHNSW_IVFPQ(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph)
    {
        utils::ScopedTimer<std::chrono::seconds> myTimer("Knn IndexHNSWFlat & IndexIVFPQ (approximate)", "sec");

        const auto m = static_cast<faiss::MetricType>(faissMetric);

        Log::info("computeIndexHNSW_IVFPQ: k = {0} and metric {1}", nn, faiss::asText(m));

        int M = 64;                 // number of neighbors used in the graph. Larger is more accurate but uses more memory
        int efConstruction = 64;    // depth of exploration at add time
        int efSearch = 128;         // depth of exploration of the search
        faiss::ClusteringParameters cl; // helper struct, only used to access value of min_points_per_centroid
        double nlist_max = std::min(100.0, static_cast<double>(cl.min_points_per_centroid) / data.getNumPoints() - 1);
        int nlist = static_cast<int>(std::max(nlist_max, std::sqrt(data.getNumPoints()))); // the number of cells
        int pq_M = 16; // number of subquantizers
        int nbits = 8; // number of bits per quantization index

        auto d = static_cast<int>(data.getNumDimensions());
        auto np = data.getNumPoints();
        auto xb = data.data();

        auto iHNSWFlat = faiss::IndexHNSWFlat(d, M, m); // create quantizer
        iHNSWFlat.hnsw.efConstruction = efConstruction;
        iHNSWFlat.hnsw.efSearch = efSearch;

        auto iIVFPQ = faiss::IndexIVFPQ(&iHNSWFlat, d, nlist, pq_M, nbits, m);    // create index

        assert(!iIVFPQ.is_trained);

        Log::info("computeIndexHNSW_IVFPQ: training...");
        iIVFPQ.train(np, xb);

        assert(iIVFPQ.is_trained);

        Log::info("computeIndexHNSW_IVFPQ: adding...");
        iIVFPQ.add(np, xb);

        Log::info("computeIndexHNSW_IVFPQ: training...");
        iIVFPQ.search(
            np,
            xb,
            nn,
            knnGraph.knnDistances.data(),
            knnGraph.knnIndices.data()
        );

    }

    std::optional<std::vector<int64_t>> checkAllNeighborsExist(const utils::GraphView knnGraphView) {

        std::vector<int64_t> incompletePoints;

        const auto& nns = knnGraphView.getNns();

        for (int64_t p = 0; p < knnGraphView.getNumPoints(); ++p)
        {
            auto distances = knnGraphView.getDistances(p);
            auto neighbors = knnGraphView.getNeighbors(p);

            if (distances.size() != neighbors.size()) {
                incompletePoints.push_back(p);
                continue;
            }

            if (distances.size() != static_cast<size_t>(nns[p])) {
                incompletePoints.push_back(p);
                continue;
            }

            if (std::any_of(neighbors.begin(), neighbors.end(), [](const auto val) { return val == -1; })) {
                incompletePoints.push_back(p);
                continue;
            }
        }

        return incompletePoints.empty() ? std::nullopt : std::optional<std::vector<int64_t>>{ incompletePoints };
    }

} // namespace sph::utils
