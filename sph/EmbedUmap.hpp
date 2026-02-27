#pragma once

#include "utils/CommonDefinitions.hpp"
#include "utils/Graph.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace umappp {
    template<typename Index_, typename Float_>
    class Status;
}

namespace sph {

    struct UmapEmbeddingParameters
    {
        uint32_t    numEpochs = 1000;
        bool        presetEmbedding = false;
        uint32_t    outputDimensions = 2;
        bool        singleStep = false;
    };

    /*! Compute a UMAP embedding
     *
     * Builds on https://github.com/libscran/umappp
     *
     * Use as:
     *  UmapComputation umap(knnGraph, params);
     *  umap.compute(numIter);
     *	const std::vector<float>& emb = umap.output();
     */
    class UmapComputation
    {
        using UMAP = umappp::Status<int64_t, float>;

    public:
        UmapComputation() = default;
        ~UmapComputation();

        UmapComputation(const utils::GraphBaseInterface* distanceGraph, const UmapEmbeddingParameters& params, const vf32& initial_embedding = vf32());
        UmapComputation(const SparseMatHDI* distanceGraph, const UmapEmbeddingParameters& params, const vf32& initial_embedding = vf32());

        /*! Computes umap embedding */
        void compute();

        void initProbabilityDistribution();
        void runGradientDescent();
        void runGradientDescentForEpochs(uint32_t epochs);

        void stop() { 
            _shouldStop = true; 
        }

        void resetStop() {
            _shouldStop = false;
        }

    public:    // Setter

        void setVerbose(bool verbose) { _verbose = verbose; }
        void setParams(UmapEmbeddingParameters params) { _params = params; }

        // sets _graph and _numPoints, resets _embedding
        void setNeighborGraph(const utils::GraphBaseInterface* graphI);

        // sets _graph and _numPoints, resets _embedding
        void setNeighborMatrix(const SparseMatHDI* probDist);

        // sets _embedding
        void setInitialEmbedding(const vf32& initial_embedding);

        void setData(const utils::GraphBaseInterface* graphI, const vf32& initial_embedding)
        {
            setNeighborGraph(graphI);
            setInitialEmbedding(initial_embedding);
        }

        void setData(const SparseMatHDI* probDist, const vf32& initial_embedding)
        {
            setNeighborMatrix(probDist);
            setInitialEmbedding(initial_embedding);
        }

    public:    // Getter

        const auto& getEmbedding() const { return _embedding; };
        const auto& getParams() const { return _params; };
        inline bool getVerbosity() const { return _verbose; }

        inline bool isUmapRunning() const { return _isUmapRunning; }

    private:
        // UMAP structures
        vf32                                _embedding = {};            /*!< Container for the embedding, used internally > */
        size_t                              _currentIteration = 0;
        std::shared_ptr<UMAP>               _status = nullptr;

        // Given input data, precomputed
        const utils::GraphBaseInterface*    _graph = nullptr;           // we do not own this
        const SparseMatHDI*                 _probDist = nullptr;        // we do not own this

        // Options
        UmapEmbeddingParameters             _params = {};               /*!< UMAP settings > */

        // Helper
        size_t                              _numPoints = 0;

        // Flags
        bool                                _verbose = true;            /*!< Controls number of print statements > */
        bool                                _isUmapRunning = false;     /*!< Returns whether the embedding process is currently ongoing> */
        volatile bool                       _shouldStop = false;        /*!< Whether computation should be interrupted > */
    };


} // namespace sph
