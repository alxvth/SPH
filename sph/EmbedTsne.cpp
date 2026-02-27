#include "EmbedTsne.hpp"

#include "utils/HDILibHelper.hpp"
#include "utils/Logger.hpp"
#include "utils/SparseMatrixAlgorithms.hpp"

#include <hdi/data/embedding.h>
#include <hdi/dimensionality_reduction/hd_joint_probability_generator.h>
#include <hdi/utils/scoped_timers.h>

#ifdef __RUNTIME_GPU__

#ifndef __APPLE__
#include <hdi/utils/glad/glad.h>
#endif 

#include <GLFW/glfw3.h>
#endif // __RUNTIME_GPU__

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sph {

    using ProbabilityGenerator = ::hdi::dr::HDJointProbabilityGenerator<float, SparseMatHDI>;

#ifdef __RUNTIME_GPU__
    static inline bool useGPU(const GradientDescentType& gdt)
    {
#ifndef __APPLE__
        return gdt == GradientDescentType::GPUcompute || gdt == GradientDescentType::GPUraster;
#else
        return gdt == GradientDescentType::GPUcompute;
#endif
    }
#endif

    TsneComputation::TsneComputation()
    {
        _probabilityDistribution = &_probabilityDistributionLocal;

        _CPU_tSNE.setLogger(&_logger);
#ifdef __RUNTIME_GPU__
        _GPGPU_tSNE.setLogger(&_logger);
#endif
    }

    TsneComputation::TsneComputation(const utils::GraphBaseInterface* knnGraphView, const TsneEmbeddingParameters& params, const std::vector<float>& initial_embedding) :
        TsneComputation()
    {
        assert(knnGraphView->getK() == static_cast<int> (params.perplexity * params.perplexity_multiplier));

        _knnGraph = knnGraphView;
        _params = params;
        _numPoints = knnGraphView->getNumPoints();

        // Set user-given initial embedding
        if (!initial_embedding.empty())
        {
            Log::info("TsneComputation::setup: Use user-provided initial embedding");
            setInitialEmbedding(initial_embedding);
        }
    }

    TsneComputation::TsneComputation(const SparseMatHDI& probDist, const TsneEmbeddingParameters& params, const std::vector<float>& initial_embedding) :
        TsneComputation()
    {
        _probabilityDistribution = &probDist;

        if (_probabilityDistribution == nullptr)
            Log::critical("TsneComputation::TsneComputation: _probabilityDistribution is nullptr");

        _numPoints = _probabilityDistribution->size();
        _params = params;
        _probDistGiven = true;

        // Set user-given initial embedding
        if (!initial_embedding.empty())
        {
            Log::info("TsneComputation::setup: Use user-provided initial embedding");
            setInitialEmbedding(initial_embedding);
        }
    }

    TsneComputation::~TsneComputation()
    {
#ifdef __RUNTIME_GPU__
        if (_offscreenBuffer != nullptr)
            _offscreenBuffer->destroyContext();
#endif
    }

    void TsneComputation::initProbabilityDistribution()
    {
        if (_knnGraph->isValid() && _knnGraph->getNumPoints() <= 0)
        {
            Log::warn("TsneComputation::initProbabilityDistribution: Number of points must be larger than 0. Returning.");
            return;
        }

        Log::info("t-SNE: Initialize probability distribution from pre-computed kNN of {0} data points", _knnGraph->getNumPoints());

        _isTsneRunning = true;
        _probabilityDistributionLocal.clear();
        _probabilityDistributionLocal.resize(_knnGraph->getNumPoints());

        double t = 0.0;
        {
            ::hdi::utils::ScopedTimer<double> timer(t);
            // _probabilityDistributionLocal won't be symmetrized here
            // later, initGradientDescent() computes a joint prob dist in _GPGPU_tSNE.initialize(...)
            ProbabilityGenerator::Parameters params;
            params._perplexity = _params.perplexity;
            utils::computeGaussianDistributions(_knnGraph->getKnnDistances(), _knnGraph->getKnnIndices(), static_cast<int>(_knnGraph->getK()), _probabilityDistributionLocal, params);
        }

        _probDistInit = true;
        _probabilityDistribution = &_probabilityDistributionLocal;
        Log::info("t-SNE: Compute probability distribution: {} seconds", t / 1000.);
    }

    void TsneComputation::initGradientDescent()
    {
        // check if both _probDistInit and _probDistGiven are false
        if (!_probDistInit && !_probDistGiven)
        {
            Log::warn("TsneComputation::initGradientDescent: First call initProbabilityDistribution() or provide a precomputed probDist. Returning");
            return;
        }

        // check if both _probDistInit and _probDistGiven are true
        if (_probDistInit && _probDistGiven)
            Log::warn("TsneComputation::initGradientDescent: Both initProbabilityDistribution() was called and a precomputed probDist was provided. Using the precomputed one.");

        float ex_factor = 4.f + _probabilityDistribution->size() / 60'000.f;
        _params.gradDescentParams._exaggeration_factor = std::clamp(ex_factor, 4.f, 20.f);

        Log::info("t-SNE: Init gradient descent");
        Log::info("t-SNE: Perplexity {}", _params.perplexity);
        Log::info("t-SNE: Exaggeration factor {} for {} iterations and then decaying for {} iterations", _params.gradDescentParams._exaggeration_factor, _params.gradDescentParams._remove_exaggeration_iter, _params.gradDescentParams._exponential_decay_iter);

#ifdef __RUNTIME_GPU__
        auto initGPUTSNE = [this]() {
            // Create a offscreen window
            _offscreenBuffer->initialize();

#ifndef __APPLE__
            if (_params.gradientDescentType == GradientDescentType::GPUraster)
                _GPGPU_tSNE.setType(GradientDescentGPU::GpgpuSneType::RASTER);
#endif

            if(_params.symmetricProbDist)
            {
                assert(utils::isSymmetric(*_probabilityDistribution));
                _GPGPU_tSNE.initializeWithJointProbabilityDistribution(*_probabilityDistribution, &_embedding, _params.gradDescentParams);
            }
            else
                _GPGPU_tSNE.initialize(*_probabilityDistribution, &_embedding, _params.gradDescentParams);
            };
#endif

        auto initCPUTSNE = [this]() {

            double theta = std::min(0.5, std::max(0.0, (_numPoints - 1000.0) * 0.00005));
            _CPU_tSNE.setTheta(theta);

            if (_params.symmetricProbDist)
            {
                assert(utils::isSymmetric(*_probabilityDistribution));
                _CPU_tSNE.initializeWithJointProbabilityDistribution(*_probabilityDistribution, &_embedding, _params.gradDescentParams);
            }
            else
                _CPU_tSNE.initialize(*_probabilityDistribution, &_embedding, _params.gradDescentParams);

            };

        _isTsneRunning = true;
        _currentIteration = 0;

#ifdef __RUNTIME_GPU__
        if (useGPU(_params.gradientDescentType))
            initGPUTSNE();
        else
#endif
            initCPUTSNE();
    }

    void TsneComputation::runGradientDescent(uint32_t iterations, bool verbose)
    {
        if (_shouldStop)
            return;

        if (iterations < 1)
        {
            Log::warn("TsneComputation::runGradientDescent: iterations must be at least 1. Returning");
            return;
        }

#ifdef __RUNTIME_GPU__
        auto gradientDescentSetup = [this]() -> bool {
            if (useGPU(_params.gradientDescentType))
            {
                if (!_offscreenBuffer->isInitialized())
                    _offscreenBuffer->initialize();

                _offscreenBuffer->bindContext();

                assert(_offscreenBuffer->isInitialized());

                return _GPGPU_tSNE.isInitialized();
            }
            else
                return _CPU_tSNE.isInitialized();
            };

        auto gradientDescentCleanup = [this]() {
            if (useGPU(_params.gradientDescentType))
                _offscreenBuffer->releaseContext();
            else
                return; // Nothing to do for CPU implementation
            };

        if (!gradientDescentSetup())
        {
            Log::warn("TsneComputation::runGradientDescent: First call initGradientDescent(). Returning");
            return;
        }
#endif

        auto singleTSNEIteration = [this]() {
#ifdef __RUNTIME_GPU__
            if (useGPU(_params.gradientDescentType))
                _GPGPU_tSNE.doAnIteration();
            else
#endif
                _CPU_tSNE.doAnIteration();
            };

        if (verbose)
            Log::info("TsneComputation::runGradientDescent: Computing gradient descent");

        _isTsneRunning = true;

        // Performs gradient descent for every iteration
        for (uint32_t it = 0; it < iterations; it++)
        {
            if (_shouldStop)
                break;

            singleTSNEIteration();
            _currentIteration++;
        }

#ifdef __RUNTIME_GPU__
        gradientDescentCleanup();
#endif // __RUNTIME_GPU__

        _isTsneRunning = false;

        if (verbose)
            Log::info("TsneComputation::runGradientDescent: {0} iterations (of {1})", iterations, _currentIteration);
    }

    void TsneComputation::compute(uint32_t iterations, bool verbose)
    {
        if (_numPoints == 1)
        {
            Log::info("TsneComputation: Only 1 point, do not embed.");
            _embedding.getContainer() = { 0.f, 0.f };
            return;
        }

        _shouldStop = false;

        if (!_probDistGiven)
            initProbabilityDistribution();

        initGradientDescent();
        runGradientDescent(iterations, verbose);
    }

    void TsneComputation::setNeighborGraph(const utils::GraphBaseInterface* knnGraphView)
    {
        _probDistGiven = false;
        _knnGraph = knnGraphView;
        _probabilityDistribution = nullptr;
        _numPoints = knnGraphView->getNumPoints();
        _embedding = ::hdi::data::Embedding<float>(2, static_cast<unsigned int> (_numPoints));
    }

    void TsneComputation::setProbabilityDistribution(const SparseMatHDI* probDist)
    {
        _probDistGiven = true;
        _probabilityDistribution = probDist;
        _knnGraph = nullptr;
        _numPoints = _probabilityDistribution->size();
        _embedding = ::hdi::data::Embedding<float>(2, static_cast<unsigned int> (_numPoints));
    }

    void TsneComputation::setInitialEmbedding(const std::vector<float>& initial_embedding)
    {
        if ((_knnGraph != nullptr && !_knnGraph->isValid()) && _probabilityDistribution == nullptr)
        {
            Log::warn("TsneComputation::setInitialEmbedding: _knnGraph or _probabilityDistribution must be given. Returning");
            return;
        }

        if (initial_embedding.size() != static_cast<size_t> (_numPoints * 2))
        {
            Log::warn("TsneComputation::setup: initial_embedding must be of size _numPoints * 2. Not using user-provided initial embedding.");
            return;
        }

        _embedding = ::hdi::data::Embedding<float>(2, static_cast<unsigned int> (_numPoints));
        _embedding.getContainer() = initial_embedding;
        _params.gradDescentParams._presetEmbedding = true;
    }

    void TsneComputation::setData(const utils::GraphBaseInterface* knnGraphView, const std::vector<float>& initial_embedding)
    {
        setNeighborGraph(knnGraphView);
        if (!initial_embedding.empty())
            setInitialEmbedding(initial_embedding);
    }

    void TsneComputation::continueGradientDescent(uint32_t iterations, bool verbose)
    {
        if (_numPoints == 1)
        {
            Log::warn("TsneComputation: Only 1 point, not continuing.");
            return;
        }

        _shouldStop = false;

        runGradientDescent(iterations, verbose);
    }

#ifdef __RUNTIME_GPU__

    void OffscreenBufferGLFW::initialize()
    {
        if (!glfwInit()) {
            throw std::runtime_error("Unable to initialize GLFW.");
        }

#ifdef __APPLE__
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // invisible - ie offscreen, window
        _offscreenWindow = glfwCreateWindow(640, 480, "", NULL, NULL);

        if (_offscreenWindow == NULL) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        bindContext();

#ifndef __APPLE__
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            glfwTerminate();
            throw std::runtime_error("Failed to initialize OpenGL context");
        }
#endif // Not __APPLE__

        _isInitialized = true;
    }

    void OffscreenBufferGLFW::bindContext()
    {
        glfwMakeContextCurrent(_offscreenWindow);
    }

    void OffscreenBufferGLFW::releaseContext()
    {
        glfwMakeContextCurrent(nullptr);
    }

    void OffscreenBufferGLFW::destroyContext()
    {
        releaseContext();
        glfwDestroyWindow(_offscreenWindow);
        glfwTerminate();
        _isInitialized = false;
    }

#endif // __RUNTIME_GPU__

} // namespace sph
