#include "EmbedUmap.hpp"

#include "utils/Logger.hpp"
#include "utils/ProgressBar.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <umappp/NeighborList.hpp>

#pragma warning(disable:4305) // umappp internal: 'initializing': truncation from 'double' to 'Float_' (options are all double internally)
#pragma warning(disable:4267) // umappp internal: conversion from 'size_t' to 'type'
#pragma warning(disable:4244) // umappp internal: conversion from 'type1' to 'type2'
#include <umappp/initialize.hpp>
#include <umappp/find_ab.hpp>
#include <umappp/Options.hpp>
#include <umappp/Status.hpp>
#pragma warning(default:4244)
#pragma warning(default:4267)
#pragma warning(default:4305)

#include <hdi/data/map_mem_eff.h>

#include <omp.h>

namespace sph {

    UmapComputation::UmapComputation::UmapComputation(const utils::GraphBaseInterface* distanceGraph, const UmapEmbeddingParameters& params, const vf32& initial_embedding)
    {
        setParams(params);
        setNeighborGraph(distanceGraph);

        if (!initial_embedding.empty())
            setInitialEmbedding(initial_embedding);
    }

    UmapComputation::UmapComputation::UmapComputation(const SparseMatHDI* distanceGraph, const UmapEmbeddingParameters& params, const vf32& initial_embedding)
    {
        setParams(params);
        setNeighborMatrix(distanceGraph);

        if (!initial_embedding.empty())
            setInitialEmbedding(initial_embedding);
    }

    UmapComputation::~UmapComputation() {

    }

    void UmapComputation::initProbabilityDistribution() {

        if (_numPoints == 1)
        {
            Log::info("UmapComputation: Only 1 point, do not embed.");
            _embedding = { 0.f, 0.f };
            return;
        }

        if (_shouldStop)
            return;

        if (_params.numEpochs < 1)
        {
            Log::warn("UmapComputation::runGradientDescent: _params.numEpochs must be at least 1. Returning");
            return;
        }

        if (_verbose)
            Log::info("UmapComputation::runGradientDescent: Computing gradient descent");

        _isUmapRunning = true;
        _status.reset();

        // Convert neighbor graph into lta-umap neighbor list
        int64_t numPoints = 0;

        umappp::Options opt;
        //opt.num_epochs    = _params.numEpochs;
        opt.initialize_method = umappp::InitializeMethod::SPECTRAL;
        //opt.num_neighbors =  ... ;   // no need to set this since we compute our own neighbors

        auto setParallel = [&opt](int64_t nPoints) {
            if (nPoints < 500)
                return;

            auto nThreads = omp_get_max_threads();
            opt.num_threads = nThreads > 0 ? nThreads : 1;

            if (nThreads >= 4)
                opt.parallel_optimization = true;
            };

        auto checkInitSetting = [this, &opt, &numPoints](const umappp::NeighborList<int64_t, float>& neighbors) {
            if (_params.presetEmbedding == true) {
                opt.initialize_method = umappp::InitializeMethod::NONE;
                return;
            }

            // length of shortest inner vector
            int64_t minNumNeighbors = std::min_element(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); })->size();

            if (std::min(numPoints, minNumNeighbors) < static_cast<int64_t>(_params.outputDimensions) + 1)
                opt.initialize_method = umappp::InitializeMethod::RANDOM;

            };


        if (_verbose)
            Log::info("UmapComputation::runGradientDescent: initializing...");

        if (_graph)
        {
            if (_verbose)
                Log::info("UmapComputation::runGradientDescent: Using _graph (convert to umappp neighbor list)");

            numPoints = _graph->getNumPoints();
            umappp::NeighborList<int64_t, float> neighbors;
            neighbors.resize(numPoints);

            setParallel(numPoints);

            utils::ProgressBar progress(numPoints, _verbose);
            for (int64_t i = 0; i < numPoints; i++)
            {
                const int64_t numK = _graph->getK(i);
                for (int64_t k = 1; k < numK; k++)
                {
                    const std::pair<int64_t, float> neighborPair = _graph->getNeighborDistanceN(i, k);
                    neighbors[i].push_back({ neighborPair.first, neighborPair.second });
                }

                progress.update();
            }
            progress.finish();

            checkInitSetting(neighbors);

            if (_verbose)
                Log::info("UmapComputation::runGradientDescent: initialize UMAP epochs...");

            _status = std::make_unique<UMAP>(umappp::initialize<int64_t, float>(neighbors, _params.outputDimensions, _embedding.data(), opt));

            _params.numEpochs = _status->num_epochs();

            if (_verbose)
            {
                auto ab = umappp::find_ab(opt.spread, opt.min_dist);
                Log::info("UmapComputation::runGradientDescent: a: {}, b: {}, epochs: {}", ab.first, ab.second, _params.numEpochs);
            }

        }
        else if (_probDist)
        {
            if (_verbose)
                Log::info("UmapComputation::runGradientDescent: Using _probDist (convert to umappp neighbor list and init embedding)");

            numPoints = _probDist->size();
            umappp::NeighborList<int64_t, float> neighbors;
            neighbors.resize(numPoints);

            setParallel(numPoints);

            // we do basically the same as umappp::initialize except for umappp::internal::neighbor_similarities
            // which was ideally already called earlier via
            // ImageHierarchy::computePreparations -> utils::normalizeKnnDistances -> computeExponentialDistributions 
            // which converts distances into probability-like similarities

            umappp::NeighborSimilaritiesOptions<float> nsopt;
            nsopt.local_connectivity = static_cast<float>(opt.local_connectivity);
            nsopt.bandwidth          = static_cast<float>(opt.bandwidth);
            nsopt.num_threads        = opt.num_threads;

            utils::ProgressBar progress(numPoints, _verbose);
            for (int64_t i = 0; i < numPoints; i++)
            {
                const auto& currentVec = _probDist->at(i);
                for (auto it = currentVec.begin(); it != currentVec.end(); it++)
                    neighbors[i].push_back({ static_cast<int64_t>(it->first), it->second });

                progress.update();
            }
            progress.finish();

            umappp::combine_neighbor_sets(neighbors, static_cast<float>(opt.mix_ratio));

            checkInitSetting(neighbors);

            // Choosing the manner of initialization.
            const irlba::Options irlba_opt = {};
            if (opt.initialize_method == umappp::InitializeMethod::SPECTRAL) {
                const bool success = umappp::normalized_laplacian(neighbors, _params.outputDimensions, _embedding.data(), irlba_opt, opt.num_threads, /*scale*/ 1.0);

                if (!success)
                    opt.initialize_method = umappp::InitializeMethod::RANDOM;
                
            }
            
            if (opt.initialize_method == umappp::InitializeMethod::RANDOM) {
                umappp::random_init(neighbors.size(), _params.outputDimensions, _embedding.data(), /*seed*/ 123456, /*scale*/ 1.0);
            }

            // Finding a good a/b pair.
            if (opt.a <= 0 || opt.b <= 0) {
                auto found = umappp::find_ab(opt.spread, opt.min_dist);
                opt.a      = found.first;
                opt.b      = found.second;
            }

            opt.num_epochs    = umappp::choose_num_epochs(_params.numEpochs, neighbors.size());
            _params.numEpochs = opt.num_epochs.value();

            if (_verbose)
                Log::info("UmapComputation::runGradientDescent: a: {}, b: {}, epochs: {}", opt.a.value(), opt.b.value(), opt.num_epochs.value());

            _status = std::make_unique<umappp::Status<int64_t, float>>(
                umappp::similarities_to_epochs<int64_t, float>(neighbors, opt.num_epochs.value(), static_cast<float>(opt.negative_sample_rate)),
                opt,
                _params.outputDimensions
            );
        }
        else
        {
            Log::error("UmapComputation::runGradientDescent: you must provide either _graph or _probDist");
            return;
        }

        _currentIteration = 0;
        _isUmapRunning    = false;
    }

    void UmapComputation::runGradientDescent()
    {
        if (_shouldStop)
            return;

        if (_verbose)
            Log::info("UmapComputation::runGradientDescent: running gradient descent...");

        _isUmapRunning = true;

        if (_params.singleStep)
        {
            _status->run(_embedding.data(), _params.numEpochs);
            _currentIteration = _params.numEpochs;
        }
        else
        {
            // Performs gradient descent for every iteration
            utils::ProgressBar progress(_params.numEpochs, _verbose);
            for (uint32_t iter = 1; iter < _params.numEpochs; iter++)
            {
                if (_shouldStop)
                    break;

                _status->run(_embedding.data(), iter);

                _currentIteration++;
                progress.update();
            }
            progress.finish();
        }

        _isUmapRunning = false;

        if (_verbose)
            Log::info("UmapComputation::runGradientDescent: {0} iterations (of {1})", _params.numEpochs, _currentIteration);
    }

    void UmapComputation::runGradientDescentForEpochs(uint32_t epochs)
    {
        if (_shouldStop)
            return;

        if (_currentIteration + epochs > _params.numEpochs) {
            Log::warn("UmapComputation::runGradientDescentForEpochs: {0} (current) +  {1} (new) would be more than originally set ({2})", _currentIteration, epochs, _params.numEpochs);
            return;
        }

        _isUmapRunning = true;

        for (uint32_t epoch = 0; epoch < epochs; epoch++)
            _status->run(_embedding.data(), static_cast<uint32_t>(++_currentIteration));

        _isUmapRunning = false;
    }

    void UmapComputation::compute()
    {
        _isUmapRunning = true;

        _shouldStop = false;

        initProbabilityDistribution();

        runGradientDescent();

        _isUmapRunning = false;
    }

    void UmapComputation::setNeighborGraph(const utils::GraphBaseInterface* graphI)
    {
        _graph                  = graphI;
        _probDist               = nullptr;
        _numPoints              = _graph->getNumPoints();
        _embedding              = vf32(_numPoints * _params.outputDimensions, 0);
        _params.presetEmbedding = false;
    }

    void UmapComputation::setNeighborMatrix(const SparseMatHDI* probDist)
    {
        _probDist               = probDist;
        _graph                  = nullptr;
        _numPoints              = _probDist->size();
        _embedding              = vf32(_numPoints * _params.outputDimensions, 0);
        _params.presetEmbedding = false;
    }

    void UmapComputation::setInitialEmbedding(const vf32& initial_embedding)
    {
        if (initial_embedding.size() != _numPoints * _params.outputDimensions) {
            Log::warn("UmapComputation::setup: initial_embedding must be of size _numPoints * 2. Not using user-provided initial embedding.");
            return;
        }

        _embedding              = initial_embedding;
        _params.presetEmbedding = true;
    }


} // namespace sph::utils
