#pragma once

#include "utils/CommonDefinitions.hpp"
#include "utils/Graph.hpp"

#include <cstdint>
#include <vector>

#include <hdi/data/embedding.h>
#include <hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h>
#include <hdi/dimensionality_reduction/tsne_parameters.h>
#include <hdi/utils/cout_log.h>

#ifdef __RUNTIME_GPU__
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"
#endif

namespace sph {

#ifdef __RUNTIME_GPU__
    class OffscreenBuffer;
    class OffscreenBufferGLFW;
#endif

    enum class GradientDescentType
    {
#ifdef __RUNTIME_GPU__
        GPUcompute,
#ifndef __APPLE__
        GPUraster,
#endif
#endif
        CPU,
    };

    // do not move these to Settings.h due to hdi::dr::TsneParameters dependency
    struct TsneEmbeddingParameters
    {
        ::hdi::dr::TsneParameters gradDescentParams;
        GradientDescentType     gradientDescentType = static_cast<GradientDescentType>(0);
        float                   perplexity = 30.f;
        int32_t                 perplexity_multiplier = 3;
        uint32_t                numIterations = 1000;
        bool                    symmetricProbDist = false;

    };

    /*! Compute a t-SNE embedding
     *
     * Builds on https://github.com/biovault/HDILib
     * When providing a GraphView, a constant number of neighbors is assumed
     * and a probability distribution is computed on that basis
     *
     * Use as:
     *  auto offW = new OffscreenBufferGLFW();
     *  TsneComputation tsne(knnGraph, params);
     *  tsne.setOffscreenBuffer(dynamic_cast<OffscreenBuffer*>(offW));
     *  tsne.compute(numIter);
     *	const std::vector<float>& emb = tsne.output();
     *  delete offW;
     */
    class TsneComputation
    {
        using GradientDescentCPU = ::hdi::dr::SparseTSNEUserDefProbabilities<float, SparseMatHDI>;
#ifdef __RUNTIME_GPU__
        using GradientDescentGPU = ::hdi::dr::GradientDescentTSNETexture;
#endif

    public:
        TsneComputation();
        // knnGraphView must provide distances that will be normalized
        TsneComputation(const utils::GraphBaseInterface* knnGraphView, const TsneEmbeddingParameters& params, const std::vector<float>& initial_embedding = std::vector<float>());
        // probDist already has to be normalized
        TsneComputation(const SparseMatHDI& probDist, const TsneEmbeddingParameters& params, const std::vector<float>& initial_embedding = std::vector<float>());
        ~TsneComputation();

        /*! Computes t-SNE embedding */
        void compute(uint32_t iterations, bool verbose = true);

        void continueGradientDescent(uint32_t iterations, bool verbose = true);
        void stop() { 
            _shouldStop = true; 
        }

        void resetStop() { 
            _shouldStop = false; 
        }

    public:    // Setter

        void setVerbose(bool verbose) { _verbose = verbose; }
        void setExaggerationIter(int exaggerationIter) { _params.gradDescentParams._remove_exaggeration_iter = exaggerationIter; }
        void setExponentialDecay(int exponentialDecay) { _params.gradDescentParams._exponential_decay_iter = exponentialDecay; }
        void setPerplexity(float perplexity) { _params.perplexity = perplexity; }
        void setParams(TsneEmbeddingParameters params) { _params = params; }

        void setNeighborGraph(const utils::GraphBaseInterface* knnGraphView);
        void setProbabilityDistribution(const SparseMatHDI* probDist);
        void setInitialEmbedding(const std::vector<float>& initial_embedding);

        void setData(const utils::GraphBaseInterface* knnGraphView, const std::vector<float>& initial_embedding);

#ifdef __RUNTIME_GPU__
        void setOffscreenBuffer(OffscreenBuffer* offscreenBuffer) { _offscreenBuffer = offscreenBuffer; }
#endif

    public:    // Getter

        const auto& getEmbedding() const { return _embedding; };
        const auto& getParams() const { return _params; };
        bool getVerbosity() const { return _verbose; }
        const auto& output() const { return _embedding.getContainer(); };

        bool isTsneRunning() const { return _isTsneRunning; }

        // default to GPU (Compute), or GPU (Raster) on Apple, or CPU when __RUNTIME_GPU__ is NOT defined
        static const GradientDescentType defaultGD = static_cast<GradientDescentType>(0);

    private:

        /*! Computes A-tSNE probability distribution */
        void initProbabilityDistribution();

        /*! Initializes GPU based gradient descent */
        void initGradientDescent();

        /*! Performs gradient descent iterations  */
        void runGradientDescent(uint32_t iterations, bool verbose = true);

    private:
        // TSNE structures
        SparseMatHDI                    _probabilityDistributionLocal = {};         /*!< Generator for a probability distribution that describes similarities in the high dimensional data > */
        GradientDescentCPU              _CPU_tSNE = {};                             /*!< CPU based t-sne computation class> */
        ::hdi::data::Embedding<float>   _embedding = {};                            /*!< Container for the embedding, used internally > */
        size_t                          _currentIteration = 0;

#ifdef __RUNTIME_GPU__
        GradientDescentGPU              _GPGPU_tSNE = {};                           /*!< Main gpu based t-sne computation class> */
        OffscreenBuffer*                _offscreenBuffer = nullptr;                 /*!< offscreen window > */
#endif

        // Given input data, precomputed
        const utils::GraphBaseInterface* _knnGraph = {};                            // we do not own this
        const SparseMatHDI*             _probabilityDistribution = nullptr;         /*!< Pointer to a probability distribution that describes similarities in the high dimensional data > */

        // Options
        TsneEmbeddingParameters         _params = {};                               /*!< T-SNE settings > */

        // Helper
        size_t                          _numPoints = 0;
        ::hdi::utils::CoutLog           _logger = {};

        // Flags
        bool                            _verbose = false;                           /*!< Controls number of print statements of HDILib> */
        bool                            _isTsneRunning = false;                     /*!< Returns whether the embedding process is currently ongoing> */
        bool                            _probDistInit = false;                      /*!< Whether initProbabilityDistribution was called> */
        bool                            _probDistGiven = false;                     /** Check if the worker was initialized with a probability distribution or data */
        volatile bool                   _shouldStop = false;                        /*!< Whether computation should be interrupted > */
    };

} // namespace sph


#ifdef __RUNTIME_GPU__

struct GLFWwindow;

namespace sph {

    class OffscreenBuffer
    {
    public:
        OffscreenBuffer() : _isInitialized(false) {}

        bool isInitialized() const { return _isInitialized; }

        /** Initialize and bind the OpenGL context associated with this buffer */
        virtual void initialize() = 0;

        /** Bind the OpenGL context associated with this buffer */
        virtual void bindContext() = 0;

        /** Release the OpenGL context associated with this buffer */
        virtual void releaseContext() = 0;

        /** Destroy the OpenGL context associated with this buffer */
        virtual void destroyContext() = 0;

    protected:
        bool _isInitialized;

    };

    class OffscreenBufferGLFW : public OffscreenBuffer
    {
    public:
        OffscreenBufferGLFW() = default;

        void initialize() override;
        void bindContext() override;
        void releaseContext() override;
        void destroyContext() override;

    private:
        GLFWwindow* _offscreenWindow = nullptr;
    };

} // namespace sph

#endif // __RUNTIME_GPU__
