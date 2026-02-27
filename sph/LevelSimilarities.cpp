#include "LevelSimilarities.hpp"

#include "ImageHierarchy.hpp"

#include "utils/Algorithms.hpp"
#include "utils/FileIO.hpp"
#include "utils/GraphNormalization.hpp"
#include "utils/HDILibHelper.hpp"
#include "utils/Logger.hpp"
#include "utils/PrintHelper.hpp"
#include "utils/ProgressBar.hpp"
#include "utils/SparseMatrixAlgorithms.hpp"
#include "utils/Statistics.hpp"

#include "utils/EuclidDistSpace.hpp"
#include "utils/GeodesicPathSpace.hpp"
#include "utils/NeighborOverlapSpace.hpp"
#include "utils/NeighborWalksBhattacharyyaSpace.hpp"
#include "utils/NeighborWalksSingleOverlapSpace.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <memory>
#include <type_traits>
#include <utility>

#if SPH_RELEASE
#include <thread>
#endif

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <nlohmann/json.hpp>

#include <Eigen/SparseCore>

#include <fmt/ranges.h>

#include <hdi/data/map_mem_eff.h>
#include <hdi/dimensionality_reduction/hd_joint_probability_generator.h>

namespace sph {

    using ProbabilityGenerator = hdi::dr::HDJointProbabilityGenerator<float, SparseMatHDI>;
    using Path = std::filesystem::path;

    ///////////////////////
    // LevelSimilarities //
    ///////////////////////
    LevelSimilarities::LevelSimilarities() :
        Cacheable("LevelSimilarities")
    {
    }

    LevelSimilarities::LevelSimilarities(const utils::Hierarchy& hierarchy, const utils::Graph& dataNnGraph, const utils::Data& data, const LevelSimilaritiesSettings& lss) :
        LevelSimilarities(hierarchy, dataNnGraph.getGraphView(), data.getDataView(), lss)
    {
    }

    LevelSimilarities::LevelSimilarities(const utils::Hierarchy& hierarchy, const utils::GraphView& dataNnGraph, const utils::DataView& data, const LevelSimilaritiesSettings& lss) :
        LevelSimilarities()
    {
        _hierarchy = &hierarchy;
        _dataKnnGraph = dataNnGraph;
        _data = data;
        _lss = lss;

        initOutput();
    }

    void LevelSimilarities::initOutput()
    {
        _distanceGraphs.resize(_hierarchy->getNumLevels());
        _probDistsAllLevels.resize(_hierarchy->getNumLevels());
        _Ks.resize(_hierarchy->getNumLevels());
        _perplexityOnLevel.resize(_hierarchy->getNumLevels());

        updateNumberOfNeighbors();
    }

    void LevelSimilarities::updateNumberOfNeighbors()
    {
        if (_lss.ks.size() == _Ks.size())
            return;

        if (_hierarchy->getNumLevels() == 0)
            return;

        assert(_lss.ks.size() > 0);
        _lss.ks.resize(_hierarchy->getNumLevels());
        assert(_lss.ks.size() == _Ks.size());

        // _lss.ks[0] is the # of data level knn
        _Ks[0] = { _lss.ks[0] };
        float dataPerplexity = (_lss.ks[0] - 1) / 3.f;
        _perplexityOnLevel[0] = std::clamp(dataPerplexity, 10.f, 100.f);

        for (size_t level = 1; level < _hierarchy->getNumLevels(); level++)
        {
            const auto& numComponentsOnLevel    = _hierarchy->numComponentsOn(level);
            float levelPerplexity               = std::clamp(numComponentsOnLevel / 100.f, 10.f, 100.f);
            levelPerplexity                     = std::min(dataPerplexity, levelPerplexity);
            //float reductionRate                 = static_cast<float>(numComponentsOnLevel) / _hierarchy->numComponentsOn(level -1);
            //float levelPerplexity               = _perplexityOnLevel[level-1] * reductionRate;
            int64_t numNeighborsOnLevel         = static_cast<int64_t>(levelPerplexity) * 3 + 1;   // 3 is perplexity multiplier, 1 is point itself
            numNeighborsOnLevel                 = std::min(numNeighborsOnLevel, static_cast<int64_t>(numComponentsOnLevel));
            _lss.ks[level]                      = numNeighborsOnLevel;
            _Ks[level]                          = { numNeighborsOnLevel };
            _perplexityOnLevel[level]           = levelPerplexity;
        }

        Log::info("LevelSimilarities::updateNumberOfNeighbors: number of neighbors on levels from bottom to top: {0}", fmt::join(_lss.ks, ", "));
    }


    void LevelSimilarities::compute(const std::optional<LevelSimilaritiesSettings>& lls)
    {
        if (lls.has_value())
            setLevelSimilaritiesSettings(lls.value());

        if (_hierarchy == nullptr)
        {
            Log::error("LevelSimilarities::compute: _hierarchy must be valid. Returning.");
            return;
        }
        if (!_dataKnnGraph.isValid())
        {
            Log::error("LevelSimilarities::compute: _dataKnnGraph must be valid. Returning.");
            return;
        }
        if (!_data.isValid())
        {
            Log::error("LevelSimilarities::compute: _data must be valid. Returning.");
            return;
        }
        if (_lss.levelToCompute >= static_cast<int64_t>(_hierarchy->getNumLevels()))
        {
            Log::error("LevelSimilarities::compute: level must be smaller _hierarchy->getNumLevels(). Returning.");
            return;
        }

        if (_lss.ks.size() == 1)
            updateNumberOfNeighbors();

        _numbersOfPoints.assign(_hierarchy->numComponents.begin(), _hierarchy->numComponents.end());

        if (!loadCache())
        {
            int64_t startLevel = _lss.levelToCompute;
            int64_t endLevel = _lss.levelToCompute + 1;

            if (_lss.levelToCompute < 0)
            {
                Log::info("LevelSimilarities::compute: Compute all levels");
                startLevel = 0;
                endLevel = static_cast<int64_t>(_hierarchy->getNumLevels());
            }

            for (_currentLevel = startLevel; _currentLevel < endLevel; _currentLevel++)
            {
                Log::info("LevelSimilarities::compute: Level {0}", _currentLevel);

                _knnIndices = &(_distanceGraphs[_currentLevel].getKnnIndices());
                _knnDistances = &(_distanceGraphs[_currentLevel].getKnnDistances());

                _knnDistances->clear();
                _knnIndices->clear();
                _Ks[_currentLevel].clear();

                computeNearestNeighborOnLevel();
                computeProbDistOnLevel();

            }
            _currentLevel--;

            symmetrizeOutput(_lss.computeSymmetricProbDist);

            writeCache();
        }
    }

    void LevelSimilarities::resetOutput() {
        _distanceGraphs.clear();
        _probDistsAllLevels.clear();
        _Ks.clear();
        _perplexityOnLevel.clear();
    };

    void LevelSimilarities::computeNearestNeighborOnLevel()
    {
        if (_currentLevel == 0)
            return;

        assert(_currentLevel >= 0);
        assert(_distanceGraphs.size() == _hierarchy->getNumLevels());
        assert(_Ks.size() == _hierarchy->getNumLevels());

        auto initKnn = [this]() {
            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Started");

            const auto numPoints = _hierarchy->numComponentsOn(_currentLevel);

            _knnIndices->resize(numPoints * _currentK, -1);
            _knnDistances->resize(numPoints * _currentK, -1);

            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: {0} points on level {1}", numPoints, _currentLevel);
            };

        auto computeExactKnn = [this](std::unique_ptr<hnswlib::SpaceInterface<float>>& searchSpace) {

            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Start exact knn computation");

            const auto numPoints = _hierarchy->numComponentsOn(_currentLevel);
            _distanceGraphs[_currentLevel].numPoints = numPoints;

            std::vector<std::pair<int64_t, float>> indices_distances(numPoints);

            hnswlib::DISTFUNC<float> distfunc = searchSpace->get_dist_func();
            void* params = searchSpace->get_dist_func_param();

            utils::ProgressBar progress(numPoints);

            // For each point, calc distances to all other
            // and take the nn smallest as kNN
            for (uint64_t i = 0; i < numPoints; i++) {

                // Calculate distance to all points using the respective metric
                SPH_PARALLEL
                for (int64_t j = 0; j < static_cast<int64_t>(numPoints); j++) {
                    utils::ComponentID compI = { static_cast<uint64_t>(_currentLevel), i };
                    utils::ComponentID compJ = { static_cast<uint64_t>(_currentLevel), static_cast<uint64_t>(j) };
                    indices_distances[j] = std::make_pair(j, distfunc(&compI, &compJ, params));
                }

                // sort all distances to point i
                std::sort(indices_distances.begin(), indices_distances.end(), [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {return a.second < b.second; });

                // Take the first _k indices and distances
                std::transform(indices_distances.begin(), indices_distances.begin() + _currentK, _knnIndices->begin() + i * _currentK, [](const std::pair<int64_t, float>& p) { return p.first; });
                std::transform(indices_distances.begin(), indices_distances.begin() + _currentK, _knnDistances->begin() + i * _currentK, [](const std::pair<int64_t, float>& p) { return p.second; });

                progress.update();
            }
            progress.finish();

            _distanceGraphs[_currentLevel].updateFixedNumNeighbors(_currentK);

            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Finished");

            };

        auto computeApproximateKnn = [this](std::unique_ptr<hnswlib::SpaceInterface<float>>& searchSpace) {

            const auto numPoints = _hierarchy->numComponentsOn(_currentLevel);
            _distanceGraphs[_currentLevel].numPoints = numPoints;

            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Start building HNSW search structure");

            // use default HNSW values for M, ef_construction
            hnswlib::HierarchicalNSW<float> appr_alg(
                searchSpace.get(),
                numPoints,
                /* M = */ 16,
                /* ef_construction = */ 200,
                /* random_seed = */ 0
            );

            // Build the search structure
            auto addHNSWPoint = [currentLevel = this->_currentLevel, &appr_alg](size_t i) -> void {
                utils::ComponentID cid(currentLevel, i);
                appr_alg.addPoint((const void*)(&cid), (hnswlib::labeltype)i);
                };

            // add the first data point outside the parallel loop
            addHNSWPoint(0);

            // init a progress bar
            utils::ProgressBar progress(numPoints);

#if SPH_RELEASE
            // This loop is for release mode, it's parallel loop implementation from hnswlib
            int num_threads = std::thread::hardware_concurrency();
            hnswlib::ParallelFor(1, numPoints, num_threads, [&](size_t i, [[maybe_unused]] size_t threadId) {
                addHNSWPoint(i);
                progress.updateSafe();
                });
#else
            for (uint64_t i = 1; i < numPoints; ++i)
            {
                addHNSWPoint(i);
                progress.update();
            }
#endif
            progress.finish();

            // Query the search structure
            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Query search structure for {0} neighbors", _currentK);
            progress.reset();

            SPH_PARALLEL
            for (int64_t i = 0; i < static_cast<int64_t>(numPoints); ++i)
            {
                // find nearest neighbors
                utils::ComponentID cid(_currentLevel, i);
                auto top_candidates = appr_alg.searchKnn((void*)(&cid), (hnswlib::labeltype)_currentK);
                while (top_candidates.size() > static_cast<size_t>(_currentK)) {
                    top_candidates.pop();
                }

                assert(top_candidates.size() == static_cast<size_t>(_currentK));

                // save nn in _knnIndices and _knnDistances 
                auto* distances_offset = _knnDistances->data() + (i * _currentK);
                auto indices_offset = _knnIndices->data() + (i * _currentK);
                int j = 0;
                while (top_candidates.size() > 0) {
                    auto& rez = top_candidates.top();
                    distances_offset[_currentK - j - 1] = rez.first;
                    indices_offset[_currentK - j - 1] = appr_alg.getExternalLabel(static_cast<hnswlib::tableint>(rez.second));
                    top_candidates.pop();
                    ++j;
                }

                SPH_PARALLEL_CRITICAL
                progress.update();
            }
            progress.finish();

            _distanceGraphs[_currentLevel].updateFixedNumNeighbors(_currentK);

            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Finished");
            };

        auto computeKnn = [this, initKnn, computeExactKnn, computeApproximateKnn](std::unique_ptr<hnswlib::SpaceInterface<float>>& searchSpace) {
            initKnn();

            if (_lss.exactKnn)
                computeExactKnn(searchSpace);
            else
                computeApproximateKnn(searchSpace);

            };

        auto useRandomWalksAsKnnDistances = [this]() {
            Log::info("LevelSimilarities::computeNearestNeighborOnLevel: Use random walks as knn distances");

            const auto& randomWalks = _hierarchy->randomWalks[_currentLevel];
            const auto numPoints = _hierarchy->numComponentsOn(_currentLevel);
            _distanceGraphs[_currentLevel].numPoints = numPoints;

            utils::ProgressBar progress(numPoints);
            for (int64_t i = 0; i < static_cast<int64_t>(numPoints); ++i)
            {
                const auto& sparseVec = randomWalks[i];

                std::vector<std::pair<int64_t, float>> denseRandomWalkPairs;

                // randomWalks contain a similarity - create distance from them
                for (int64_t i = 0; i < sparseVec.size(); ++i) {
                    if (sparseVec.coeff(i) != 0)
                        denseRandomWalkPairs.push_back(std::make_pair(i, 1.f - sparseVec.coeff(i)));
                    
                }

                std::stable_sort(SPH_PARALLEL_EXECUTION
                    denseRandomWalkPairs.begin(), 
                    denseRandomWalkPairs.end(),
                    [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {
                        return a.second < b.second;
                    });

                for (const auto& pair : denseRandomWalkPairs) {
                    _knnIndices->push_back(pair.first);
                    _knnDistances->push_back(pair.second);
                }

                _Ks[_currentLevel].push_back(denseRandomWalkPairs.size());
                _distanceGraphs[_currentLevel].nns.push_back(denseRandomWalkPairs.size());
                progress.update();
            }
            _distanceGraphs[_currentLevel].updateOffsets();

            progress.finish();

            assert(_distanceGraphs[_currentLevel].isValid());

            };

        // Set number of current nearest neighbors
        const auto numPoints = _hierarchy->numComponentsOn(_currentLevel);
        _currentK = _lss.ks[_currentLevel];
        if (_currentK > static_cast<int64_t>(numPoints))
        {
            Log::info("LevelSimilarities::computeNearestNeighborOnLevel:: fewer number of points ({0}) than requested neighbors ({1}), setting current k to numPoints", numPoints, _currentK);
            _currentK = static_cast<int64_t>(numPoints);
        }
        _Ks[_currentLevel] = { _currentK };

        // Use different neighborhood similarities depending on setting
        std::unique_ptr<hnswlib::SpaceInterface<float>> searchSpace;
        switch (_lss.componentSim)
        {
        case utils::ComponentSim::GEO_CENTROID:
            searchSpace = std::make_unique<hnswlib::GeodesicPathSpace>(_hierarchy, &_dataKnnGraph, _data, _lss.componentLabels);
            computeKnn(searchSpace);
            break;
        case utils::ComponentSim::NEIGH_OVERLAP:
            searchSpace = std::make_unique<hnswlib::NeighborOverlapSpace>(_hierarchy, &_dataKnnGraph);
            computeKnn(searchSpace);
            break;
        case utils::ComponentSim::EUCLID_CENTROID:
            searchSpace = std::make_unique<hnswlib::EuclidDistSpace>(_hierarchy, _data);
            computeKnn(searchSpace);
            break;
        case utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP:

            if (_hierarchy->settings.rwHandling == utils::RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN)
            {
                searchSpace = std::make_unique<hnswlib::NeighborWalksSingleOverlapSpace>(_hierarchy);
                computeKnn(searchSpace);
            }
            else if (_lss.forceComputeDistances)
                useRandomWalksAsKnnDistances();

            break;
        case utils::ComponentSim::GEO_WALKS: [[fallthrough]];
        case utils::ComponentSim::NEIGH_WALKS:

            if (_hierarchy->settings.rwHandling == utils::RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN)
            {
                searchSpace = std::make_unique<hnswlib::NeighborWalksBhattacharyyaSpace>(_hierarchy);
                computeKnn(searchSpace);
            }
            else if (_lss.forceComputeDistances)
                useRandomWalksAsKnnDistances();

            break;
        }

    }

    void LevelSimilarities::computeProbDistOnLevel()
    {
        Log::info("LevelSimilarities::computeProbDist: Start computing");
        assert(_currentLevel >= 0);
        assert(_probDistsAllLevels.size() == _hierarchy->getNumLevels());

        const auto numPoints = _hierarchy->numComponentsOn(_currentLevel);

        // the probability distribution won't be symmetric here
        auto& probDistOnLevel = _probDistsAllLevels[_currentLevel];
        probDistOnLevel.clear();
        probDistOnLevel.resize(numPoints);

        _stats.perplexities.push_back(_perplexityOnLevel[_currentLevel]);
        _stats.ks.push_back(_currentK);

        auto useRandomWalks = [this, &probDistOnLevel, &numPoints]() {
            Log::info("LevelSimilarities::computeProbDist: Use random walks as probability distribution (largest {0} entries)", _currentK);

            if (_lss.randomWalkPairSims)
            {
                Log::info("LevelSimilarities::computeProbDist: compute distances based on pairs of random walk distribution");

                std::optional<std::reference_wrapper<const vvui64>> transitionWeights = std::nullopt;
                if (_lss.weightTransitionBySize)
                    transitionWeights = _hierarchy->mapFromLevelToPixel[_currentLevel];

                const int verbosityLevel = 1;

                utils::printSparseMatrixStats(utils::sparseMatrixStats(_hierarchy->randomWalks[_currentLevel]), "randomWalks");
                
                probDistOnLevel = utils::createSimilarities(_hierarchy->randomWalks[_currentLevel], _currentK, 0.0001f, verbosityLevel, transitionWeights);
            }
            else
            {
                Log::info("LevelSimilarities::computeProbDist: use random walk probability distribution");
                // higher values mean larger similarity since the overlap
                const SparseMatSPH& randomWalks = _hierarchy->randomWalks[_currentLevel];

                Log::info("LevelSimilarities::computeProbDist: convert to HDILib data structure");
                SPH_PARALLEL
                for (int64_t i = 0; i < static_cast<int64_t>(numPoints); ++i)
                    utils::convertEigenSparseVecToHDILibSparseVec(randomWalks[i], _currentK, probDistOnLevel[i]);

            }

            utils::printSparseMatrixStats(utils::sparseMatrixStats(probDistOnLevel), "probDistOnLevel");

            // ignore: no need to ignore points themselves since they were removed earlier -> -1
            if(_lss.normalizeProbDist == utils::NormalizationScheme::TSNE)
            {
                Log::info("LevelSimilarities::computeProbDist: Gaussian norm random walk probability distribution (with perplexity {0}) for t-SNE", _perplexityOnLevel[_currentLevel]);
                // perplexity: use nn of each point instead of _perplexityOnLevel[_currentLevel] since often there are fewer values
                utils::computeGaussianDistributions(probDistOnLevel, _perplexityOnLevel[_currentLevel], -1);
            }
            else if (_lss.normalizeProbDist == utils::NormalizationScheme::UMAP)
            {
                Log::info("LevelSimilarities::computeProbDist: Exponential norm random walk probability distribution for UMAP");
                utils::computeExponentialDistributions(probDistOnLevel, -1);
            }
            else
                Log::warn("LevelSimilarities::computeProbDist: normalizeProbDist should be TSNE or UMAP, it is neither. No normalization...");

            utils::printSparseMatrixStats(utils::sparseMatrixStats(probDistOnLevel), "probDistOnLevel (after norm)");
            };

        auto useKnnDistances = [this, &probDistOnLevel]() {
            ProbabilityGenerator::Parameters params;
            params._perplexity = _perplexityOnLevel[_currentLevel];
            Log::info("LevelSimilarities::computeProbDist: Normalize knn distances (compute gaussian distributions) with perplexity {}", params._perplexity);
            utils::computeGaussianDistributions(*_knnDistances, *_knnIndices, static_cast<int>(_currentK), probDistOnLevel, params);
            };

        if (_currentLevel == 0)
        {
            if (_imgHierarchy != nullptr)
            {
                Log::info("LevelSimilarities::computeProbDist: using probability distribution form image hierarchy");
                auto& dataProbDist = _imgHierarchy->getDataLevelProbdist();

                utils::convertSparseMatSPHToSparseMatHDI(dataProbDist, probDistOnLevel);
            }
            else
            {
                Log::info("LevelSimilarities::computeProbDist: No image hierarchy given, compute {} norm of data graph", _lss.normalizeProbDist);
                
                if (_lss.normalizeProbDist == utils::NormalizationScheme::TSNE)
                    utils::computeGaussianDistributions(_dataKnnGraph, probDistOnLevel, _perplexityOnLevel[_currentLevel]);
                else if (_lss.normalizeProbDist == utils::NormalizationScheme::UMAP)
                    utils::computeExponentialDistributions(_dataKnnGraph, probDistOnLevel);
                else
                    Log::warn("LevelSimilarities::computeProbDist: normalizeProbDist should be TSNE or UMAP, it is neither. No normalization...");

                utils::printSparseMatrixStats(utils::sparseMatrixStats(probDistOnLevel), "probDistOnLevel (after norm)");
            }
        }
        else
        {
            switch (_lss.componentSim)
            {
            case utils::ComponentSim::GEO_WALKS: [[fallthrough]];
            case utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP: [[fallthrough]];
            case utils::ComponentSim::NEIGH_WALKS:

                if (_hierarchy->settings.rwHandling == utils::RandomWalkHandling::MERGE_RW_ONLY ||
                    _hierarchy->settings.rwHandling == utils::RandomWalkHandling::MERGE_RW_NEW_WALKS ||
                    _hierarchy->settings.rwHandling == utils::RandomWalkHandling::MERGE_DATA_NEW_WALKS)
                    useRandomWalks();
                else if (_hierarchy->settings.rwHandling == utils::RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN)
                    useKnnDistances();
                else
                    Log::error("LevelSimilarities::computeProbDist: unhandled settings.rwHandling case");

                break;
            case utils::ComponentSim::GEO_CENTROID: [[fallthrough]];
            case utils::ComponentSim::EUCLID_CENTROID: [[fallthrough]];
            case utils::ComponentSim::NEIGH_OVERLAP:
                useKnnDistances();
                break;
            }
        }

        // Remove zero values (e.g. in case less than specified knn where found)
        for (auto& proDistForPoint : probDistOnLevel)
        {
            auto& proDistForPointStorage = proDistForPoint.memory();

            proDistForPointStorage.erase(
                std::remove_if(
                    proDistForPointStorage.begin(),
                    proDistForPointStorage.end(),
                    [](const auto& p) {
                        return p.second == 0.0f;
                    }
                ),
                proDistForPointStorage.end()
            );
        }

        const auto stats = utils::sparseMatrixStats(probDistOnLevel);
        _stats.avgNumNeighbors.push_back(static_cast<float>(stats.averageNonZeros));

        Log::info("LevelSimilarities::computeProbDist: Finished (probdist is not symmetric)");
    }

    void LevelSimilarities::symmetrizeOutput(utils::NormalizationScheme method)
    {
        if (method == utils::NormalizationScheme::NONE)
            return;

        if (_probDistIsSymmetric != utils::NormalizationScheme::NONE)
        {
            Log::info("LevelSimilarities::symmetrizeOutput: already symmetric for {}", method);
            return;
        }

        if (_lss.normalizeProbDist != method)
        {
            Log::info("LevelSimilarities::symmetrizeOutput: probDist is normalized with {}. Won't apply symmetrization for {}", _lss.normalizeProbDist, method);
            return;
        }

        Log::info("LevelSimilarities::symmetrizeOutput: {}", method);

        for (auto& probDistOnLevel : _probDistsAllLevels)
        {
            if (method == utils::NormalizationScheme::UMAP)
                utils::symmetrizeUMAP(probDistOnLevel);
            else if (method == utils::NormalizationScheme::TSNE)
                utils::symmetrizeTSNE(probDistOnLevel);
            else
            {
                Log::info("LevelSimilarities::symmetrizeOutput: method must be NormalizationScheme::TSNE or NormalizationScheme::UMAP");
                return;
            }
        }

        _probDistIsSymmetric = method;
        Log::info("LevelSimilarities::symmetrizeOutput: finished");
    }

    bool LevelSimilarities::writeStats(const std::string& fileName) const {

        Log::info("LevelSimilarities::writeStats: write to {}", fileName);

        // store parameters in json file
        nlohmann::json parameters;

        parameters["perplexities"] = _stats.perplexities;
        parameters["ks"] = _stats.ks;
        parameters["avgNumNeighbors"] = _stats.avgNumNeighbors;

        // Write to file
        bool successWritingParameters = utils::writeJsonToDisk(fileName, parameters);

        if (!successWritingParameters)
            return false;

        return true;
    }

    /// /////// ///
    /// Caching ///
    /// /////// ///

    bool LevelSimilarities::checkCacheParameters(const std::string& fileName) const {
        auto [parameters, successLoadingParameters] = utils::loadJsonFromDisk(fileName);

        if (!successLoadingParameters)
            return false;

        if (!isVersionCompatible(parameters))
            return false;

        if (!utils::checkEntry(std::string("Input data name"), parameters, _cacheFileName)) return false;
        if (!utils::checkEntry("Number of points", parameters, _data.getNumPoints())) return false;
        if (!utils::checkEntry("Number of hierarchy levels", parameters, _hierarchy->getNumLevels())) return false;

        if (!utils::checkSettings(parameters, _lss)) return false;

        Log::info("Parameters of cache correspond to current settings.");

        return true;
    }

    bool LevelSimilarities::writeCacheParameters(const std::string& fileName) const
    {
        Log::info("LevelSimilarities::writeCacheParameters: write to " + fileName);

        // store parameters in json file
        nlohmann::json parameters;
        parameters["## VERSION ##"] = _cacheParameterVersion;

        parameters["Input data name"] = _cacheFileName;
        parameters["Number of points"] = _dataKnnGraph.getNumPoints();
        parameters["Number of hierarchy levels"] = _hierarchy->getNumLevels();

        utils::addToJson(_lss, parameters);

        // Write to file
        bool successWritingParameters = utils::writeJsonToDisk(fileName, parameters);

        if (!successWritingParameters)
            return false;

        return true;
    }

    bool LevelSimilarities::loadCacheSimilarities(const std::string& fileNameBase)
    {
        _distanceGraphs.clear();
        _distanceGraphs.resize(_hierarchy->getNumLevels());

        bool success = false;
        std::string fileName = "";

        auto checkLoad = [](bool s, int64_t l, const std::string& f) -> bool {
            if (!s)
            {
                Log::warn("Loading failed: at level {0} for file {1}", l, f);
                return false;
            }
            return true;
            };

        for (int64_t level = 0; level < static_cast<int64_t>(_hierarchy->getNumLevels()); level++)
        {
            fileName = fileNameBase + "_lsGraph_" + std::to_string(level) + ".cache";
            Log::info("LevelSimilarities::loadCacheSimilarities: loading " + fileName);
            success = utils::loadCompressedGraphFromBinary(fileName, _distanceGraphs[level]);
            if (!checkLoad(success, level, fileName)) break;
        }

        _currentLevel = _hierarchy->getNumLevels() - 1;
        assert(_currentLevel >= 0);

        _knnIndices = &(_distanceGraphs[_currentLevel].getKnnIndices());
        _knnDistances = &(_distanceGraphs[_currentLevel].getKnnDistances());

        return success;
    }

    bool LevelSimilarities::writeCacheSimilarities(const std::string& fileNameBase) const
    {
        bool success = false;
        std::string fileName = "";

        auto checkWrite = [](bool s, int64_t l, const std::string& f) -> bool {
            if (!s)
            {
                Log::warn("Writing failed: at level {0} for file {1}", l, f);
                return false;
            }
            return true;
            };

        for (int64_t level = 0; level < static_cast<int64_t>(_hierarchy->getNumLevels()); level++)
        {
            fileName = fileNameBase + "_lsGraph_" + std::to_string(level) + ".cache";
            Log::info("LevelSimilarities::writeCacheSimilarities: writing " + fileName);
            success = utils::writeCompressedGraphToBinary(fileName, _distanceGraphs[level].getGraphView());
            if (!checkWrite(success, level, fileName)) break;
        }

        return success;
    }

    bool LevelSimilarities::loadCacheProbDist(const std::string& fileNameBase)
    {
        _probDistsAllLevels.clear();
        _probDistsAllLevels.resize(_hierarchy->getNumLevels());

        std::string fileName = "";

        for (int64_t level = 0; level < static_cast<int64_t>(_hierarchy->getNumLevels()); level++)
        {
            fileName = fileNameBase + std::to_string(level) + ".cache";
            Log::info("LevelSimilarities::loadCacheProbDist: loading " + fileName);

            bool success = utils::loadCompressedSparseMatHDIFromBinary(fileName, _probDistsAllLevels[level]);

            if (!success)
            {
                Log::warn("Loading failed: for file {}", fileName);
                return false;
            }

        }

        _probDistIsSymmetric = _lss.computeSymmetricProbDist;

        return true;
    }

    bool LevelSimilarities::writeCacheProbDist(const std::string& fileNameBase) const
    {
        std::string fileName = "";

        for (int64_t level = 0; level < static_cast<int64_t>(_hierarchy->getNumLevels()); level++)
        {
            fileName = fileNameBase + std::to_string(level) + ".cache";
            Log::info("LevelSimilarities::writeCacheProbDist: writing " + fileName);

            bool success = utils::writeCompressedSparseMatHDIToBinary(fileName, _probDistsAllLevels[level]);

            if (!success)
                Log::warn("Writing failed: for file {}", fileName);
        }

        return true;
    }

    bool LevelSimilarities::loadCacheKs(const std::string& fileNameBase)
    {
        std::string fileName = fileNameBase + "Ks.cache";
        Log::info("LevelSimilarities::loadCacheKs: writing " + fileName);

        bool success = utils::loadCompressedVecOfVecFromBinary(fileName, _Ks);

        if (!success)
            Log::warn("Loading failed: for file {}", fileName);

        return success;
    }

    bool LevelSimilarities::writeCacheKs(const std::string& fileNameBase) const
    {
        std::string fileName = fileNameBase + "Ks.cache";
        Log::info("LevelSimilarities::writeCacheKs: writing " + fileName);

        bool success = utils::writeCompressedVecOfVecToBinary(fileName, _Ks);

        if (!success)
            Log::warn("Writing failed: for file {}", fileName);

        return success;
    }

    bool LevelSimilarities::loadCache()
    {
        if (!_cacheIsActive)
        {
            Log::info("LevelSimilarities::loadCache: Caching is not active. Use setCachingActive(true) if desired.");
            return cachingFailure();
        }

        if (!cacheDependencyIsValid())
        {
            Log::info("ImageHierarchy::loadCache: Dependency not loaded from cache.");
            return cachingFailure();
        }

        Log::info("LevelSimilarities::loadCache: attempt to load cache from " + _cachePath.string());

        auto fullPath = (_cachePath / _cacheFileName).string() + "_lls";
        auto pathParameter = fullPath + "_parametersLevelSim.cache";
        auto pathKnn = fullPath + "_knn";
        auto pathProbDist = fullPath + "_probDist";
        auto pathKs = fullPath + "_Ks";

        if (!checkCacheParameters(pathParameter))
        {
            Log::warn("Loading cache failed: Current settings are different from cached parameters.");
            return cachingFailure();
        }

        if (!loadCacheSimilarities(pathKnn))
        {
            Log::warn("Loading cache failed: " + pathKnn);
            return cachingFailure();
        }

        if (!loadCacheProbDist(pathProbDist))
        {
            Log::warn("Loading cache failed: " + pathProbDist);
            return cachingFailure();
        }

        if (!loadCacheKs(pathKs))
        {
            Log::warn("Loading cache failed: " + pathKs);
            return cachingFailure();
        }

        Log::info("LevelSimilarities::loadCache: finished");
        return cachingSuccess();
    }

    bool LevelSimilarities::writeCache() const
    {
        if (!mayCache("LevelSimilarities"))
            return false;

        Log::info("LevelSimilarities::writeCache: save cache to " + _cachePath.string());

        auto fullPath = (_cachePath / _cacheFileName).string() + "_lls";
        auto pathParameter = fullPath + "_parametersLevelSim.cache";
        auto pathKnn = fullPath + "_knn";
        auto pathProbDist = fullPath + "_probDist";
        auto pathKs = fullPath + "_Ks";

        bool successParam = writeCacheParameters(pathParameter);
        bool successKnn = writeCacheSimilarities(pathKnn);
        bool successProbDist = writeCacheProbDist(pathProbDist);
        bool successKs = writeCacheKs(pathKs);

        Log::info("LevelSimilarities::writeCache: finished");
        return successParam && successKnn && successProbDist && successKs;
    }

} // namespace sph
