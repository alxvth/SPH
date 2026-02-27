#include "ImageHierarchy.hpp"

#include "utils/Algorithms.hpp"
#include "utils/CommonDefinitions.hpp"
#include "utils/FileIO.hpp"
#include "utils/GraphBoost.hpp"
#include "utils/GraphNormalization.hpp"
#include "utils/GraphUtils.hpp"
#include "utils/ImageHelper.hpp"
#include "utils/Logger.hpp"
#include "utils/Math.hpp"
#include "utils/PrintHelper.hpp"
#include "utils/ProgressBar.hpp"
#include "utils/Similarities.hpp"
#include "utils/SparseMatrixAlgorithms.hpp"
#include "utils/Statistics.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>

#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <range/v3/view/zip.hpp>

namespace sph {

    using Path = std::filesystem::path;

    ImageHierarchy::ImageHierarchy() :
        utils::Cacheable("ImageHierarchy")
    {
    }

    ImageHierarchy::ImageHierarchy(const utils::Graph& dataKnnGraph, const utils::Data& data, int64_t rows, int64_t cols, bool graphHasWCC) :
        ImageHierarchy(dataKnnGraph.getGraphView(), data.getDataView(), rows, cols, graphHasWCC)
    {
    }

    ImageHierarchy::ImageHierarchy(const utils::GraphView& dataKnnGraph, const utils::DataView& data, int64_t rows, int64_t cols, bool graphHasWCC) :
        ImageHierarchy()
    {
        _dataKnnGraph = dataKnnGraph;
        _data = data;
        _numRows = rows;
        _numCols = cols;
        _dataKnnWCC = graphHasWCC;

        assert(_data.getData().size() == static_cast<size_t>(rows * cols * _data.getNumDimensions()));
        assert(_data.getNumPoints() == rows * cols);
    }

    void ImageHierarchy::setRandomWalkSettings(const utils::RandomWalkSettings& rws) {
        _rws = rws;

        if (_rws.singleWalkLength < _rws.minimumSingleWalkLength) {
            Log::warn("ImageHierarchy::setRandomWalkSettings: _rws.singleWalkLength < _rws.minimumSingleWalkLength -> adjusting _rws.minimumSingleWalkLength");
            _rws.minimumSingleWalkLength = _rws.singleWalkLength;
        }

    };

    std::filesystem::path ImageHierarchy::getGeoCacheFile() const
    {
        auto nbKnn = static_cast<uint32_t>(_dataKnnGraph.getK());
        auto nbSim = static_cast<uint32_t>(_dataKnnGraph.isSymmetric());
        auto nbWCC = _dataKnnWCC;

        std::string ccFolder = "nbKnn" + std::to_string(nbKnn) + "nbSim" + std::to_string(nbSim) + "nbWCC" + std::to_string(nbWCC);
        auto ccPath = std::filesystem::path(_geoCache.path) / ccFolder;

        return ccPath;
    }

    bool ImageHierarchy::checkGeoCachePath(const std::filesystem::path& p) const
    {
        bool res = false;

        if (!_geoCache.cacheActive)
            return res;

        std::string fileNameBase = std::filesystem::path(_geoCache.fileName).stem().string();
        if (!utils::stringContains(_geoCache.path, fileNameBase))
            return res;

        res = std::filesystem::exists(p);
        if (!res)
            res = std::filesystem::create_directories(p);

        return res;
    }

    void ImageHierarchy::updateHierarchySettings()
    {
        _hierarchy.setSettings({
            { _numCols, _numRows, _neighConnection },
            _ihs.componentSim,
            _ihs.rwNormSim,
            _ihs.rwWeightMergeBySize,
            _ihs.rwHandling,
            _ihs.rwRemoveSelfSimAfterMerging,
            _ihs.normKnnDistances,
            _ihs.numGeodesicSamples,
            _ihs.verbose
        });
    }

    void ImageHierarchy::compute(const std::optional<ImageHierarchySettings>& ihs, const std::optional<utils::RandomWalkSettings>& rws)
    {
        _hierarchy.clear();
        assert(_data.getNumPoints() == _dataKnnGraph.getNumPoints());

        // specify neighborhood and similarity functions
        if (ihs.has_value())
            setImageHierarchySettings(ihs.value());

        if (rws.has_value())
            setRandomWalkSettings(rws.value());

        updateHierarchySettings();

        Log::info("ImageHierarchy::compute: Image Hierarchy with {0} spatial neighbors and up to {1} minimum number of components", _ihs.neighborConnection, _ihs.minNumComp);
        Log::info("ImageHierarchy::compute: Using {} similarity", _ihs.componentSim);
        Log::info("ImageHierarchy::compute: {0} minimum similarity interpreted as {1} and we are {2}merging multiple", _ihs.maxDist, (_ihs.usePercentile) ? "percentile" : "absolute value", (_ihs.mergeMultiple) ? "" : "NOT ");
        Log::info("ImageHierarchy::compute: Min reduction rate is {0}%", _ihs.minReduction);
        Log::info("ImageHierarchy::compute: Given data has {0} points and the data knn is of size {1}", _data.getNumPoints(), _dataKnnGraph.getKnnDistances().size());

        if (!loadCache())
        {
            Log::info("ImageHierarchy::compute: starting");

            // some similarities require prep-work
            computePreparations();

            // compute image hierarchy
            computeBoruvkaHierarchy();

            writeCache();
        }

        utils::Logger::flush();
    }

    void ImageHierarchy::computePreparations()
    {
        Log::info("ImageHierarchy::computePreparations: Normalize nearest neighbor distances with {0}", _ihs.normKnnDistances);

        utils::normalizeKnnDistances(_dataKnnGraph, _ihs.normKnnDistances, _dataLevelProbdist);

        // for level 0 all pixels are individual components
        const int64_t numPoints = _data.getNumPoints();
        _hierarchy.clear();
        _hierarchy.initFirstLevel(numPoints);

        switch (getComponentSim())
        {
        case utils::ComponentSim::GEO_WALKS: [[fallthrough]];
        case utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP: [[fallthrough]];
        case utils::ComponentSim::NEIGH_WALKS:
        {
            Log::info("ImageHierarchy::computePreparations: Compute random walks on data level");

            // random walks, remove diagonal elements and normalize
            auto& randomWalksOnData = _hierarchy.randomWalks.emplace_back();
            utils::SparseMatrixStats randomWalkStats;
            _ihs.rwRandomWalkLengths.push_back(_rws.singleWalkLength);
            utils::doRandomWalks(_dataLevelProbdist, _rws, randomWalksOnData, randomWalkStats, _ihs.verbose);

            _stats.rwSparsities.push_back(randomWalkStats.sparsityEffective);

            // Save data distances for later merging in hierarchy
            if (_ihs.rwHandling == utils::RandomWalkHandling::MERGE_DATA_NEW_WALKS)
                _hierarchy.mergedDataGraphs.push_back(std::make_unique<utils::GraphView>(_dataKnnGraph));

            break;
        }
        case utils::ComponentSim::NEIGH_OVERLAP: [[fallthrough]];
        case utils::ComponentSim::GEO_CENTROID: [[fallthrough]];
        case utils::ComponentSim::EUCLID_CENTROID: [[fallthrough]];
        default:
            // do nothing
            break;
        }

    }

    utils::Graph ImageHierarchy::computeDistances(uint64_t numComponentsCurrent, uint64_t currentLevel)
    {
        utils::Graph distances = {};
        distances.numPoints = numComponentsCurrent;

        // compute spatial neighbor ids
        {
            utils::ComponentID currentID{};
            currentID.level = currentLevel;

            // get neighbors and create data structure. On higher levels the number of neighbors will vary due to the component merging
            for (int64_t compID = 0; compID < static_cast<int64_t>(numComponentsCurrent); ++compID)
            {
                currentID.id = static_cast<uint64_t>(compID);
                appendNode(distances, getSpatialNeighbors(currentID));
            }
        }

        distances.updateOffsets();
        assert(distances.isValid());

        if ((getComponentSim() == utils::ComponentSim::GEO_WALKS || getComponentSim() == utils::ComponentSim::GEO_CENTROID) && !_hierarchy.bgraph)
            Log::warn("ImageHierarchy::computeComponentDistance: Boost helper graph not initialized, falling back to own A* implementation");

        // compute distances between spatial neighbors
        utils::ProgressBar progress(numComponentsCurrent);

        SPH_PARALLEL
        for (int64_t compID = 0; compID < static_cast<int64_t>(numComponentsCurrent); ++compID)
        {
            std::span<float> dists      = distances.getDistancesRef(compID);
            std::span<int64_t> neighs   = distances.getNeighborsRef(compID);

            assert(dists.size() == neighs.size());

            const utils::ComponentID currentID = { currentLevel, static_cast<uint64_t>(compID) };
            utils::ComponentID neighborID = { currentLevel, /*id*/ 0 }; // set to neighbors in following loop

            for (const auto& [neighID, dist] : ranges::views::zip(neighs, dists))
            {
                neighborID.id = static_cast<uint64_t>(neighID);
                dist = utils::componentDistance(getComponentSim(), _hierarchy, &_dataKnnGraph, _data, currentID, neighborID, _ihs.componentLabels);
            }

            utils::synchronizedSort(dists, neighs, std::less<float>{});
            assert(dists.front() == 0);
            assert(neighs.front() == compID);

            SPH_PARALLEL_CRITICAL
            progress.update();
        }

        progress.finish();

        assert(distances.isValid());

        return distances;
    }

    inline static uint64_t RandomMergeNeighbor(uint64_t compID, const std::span<const int64_t>& locSpNeighIDs, uint64_t currentLevel, uint64_t& forcedMergeCounter) {
        forcedMergeCounter++;
        const auto randomIndex = utils::randomNumberUniform(1, locSpNeighIDs.size() - 1);
        uint64_t mergeNeigh = locSpNeighIDs[randomIndex];
#ifdef SPH_DEBUG
        Log::trace("ImageHierarchy::computeBoruvkaHierarchy: sim is 0 for level {0}, compID {1}. Use locSpNeighIDs[0]: {2})", currentLevel, compID, mergeNeigh);
#endif
        return mergeNeigh;
        };

    void ImageHierarchy::mergeAllBelow(float thresh, uint64_t numComponentsCurrent, uint64_t currentLevel, const utils::Graph& distances, utils::BoostGraph& levelGraph, uint64_t& zeroSimCounter, uint64_t& forcedMergeCounter) const
    {
        Log::info("ImageHierarchy::connectMostSimilarComponents: merge all neighbors below distance of {0}", thresh);
        // This is strictly speaking NOT how Boruvka works!

        std::vector<uint64_t> mergeNeighs;

        utils::ProgressBar progress(numComponentsCurrent);
        for (const auto& compID : std::views::iota(0ull, numComponentsCurrent))
        {
            const auto locSpNeighIDs = distances.getNeighbors(compID);
            mergeNeighs.clear();
            mergeNeighs.reserve(locSpNeighIDs.size());

            const auto dists = distances.getDistances(compID);
            const auto neighs = distances.getNeighbors(compID);

            for (const auto& [neighID, dist] : ranges::views::zip(neighs, dists) | std::views::drop(1)) // we drop the self-distance which is always 0
            {
                if (dist < thresh)       // < not <=
                    mergeNeighs.push_back(neighID);
                else
                    break; // dists are sorted from smaller to larger, as soon as one value is larger, none of the following can be smaller again

            }

            // if no sims found:
            //  -> random neighbor if merging is forced, -> self which means no merging
            if (mergeNeighs.size() == 0)
            {
                zeroSimCounter++;

                if (_ihs.isAlwaysMerge())
                    mergeNeighs = { RandomMergeNeighbor(compID, locSpNeighIDs, currentLevel, forcedMergeCounter) };
                else
                {
                    Log::debug("Level {0}: component {1} has no distance", currentLevel, compID);
                    mergeNeighs = { compID };
                }
            }

            for (const auto& mergeNeigh : mergeNeighs)
                boost::add_edge(compID, mergeNeigh, levelGraph);

            Log::trace("Level {0}: merge {1} to {2}", currentLevel, compID, fmt::join(mergeNeighs, ", "));

            progress.update();
        }
        progress.finish();
    }

    void ImageHierarchy::mergeMinBelow(float thresh, uint64_t numComponentsCurrent, uint64_t currentLevel, const utils::Graph& distances, utils::BoostGraph& levelGraph, uint64_t& zeroSimCounter, uint64_t& forcedMergeCounter) const
    {
        Log::info("ImageHierarchy::connectMostSimilarComponents: merge single neighbor with min distance below {0}", thresh);

        float dist_max = std::numeric_limits<float>::max();;

        if (_ihs.verbose)
        {
            Log::debug("ImageHierarchy:: distances on level {0}", currentLevel);
            utils::printGraphAsDenseMatrix(distances, true);
        }

        utils::ProgressBar progress(numComponentsCurrent);
        for (const auto& compID : std::views::iota(0ull, numComponentsCurrent))
        {
            uint64_t mergeNeigh = compID; // init self, default is no merging
            dist_max = thresh;

            const auto dists = distances.getDistances(compID);
            const auto neighs = distances.getNeighbors(compID);

            for (const auto& [neighID, dist] : ranges::views::zip(neighs, dists) | std::views::drop(1)) // we drop the self-distance which is always 0
            {
                if (dist < dist_max)   // < not <=
                {
                    dist_max = dist;
                    mergeNeigh = neighID;
                    break; // distances are sorted from smaller to larger 
                }
            }

            if (dist_max == std::numeric_limits<float>::max())
            {
                zeroSimCounter++;

                // force merging with random neighbor, if _minSim  == -1.f and the current max sim is zero 
                if (_ihs.isAlwaysMerge())
                    mergeNeigh = RandomMergeNeighbor(compID, neighs, currentLevel, forcedMergeCounter);
                else
                    Log::debug("Level {0}: component {1} has no similarity", currentLevel, compID);
            }

            if (mergeNeigh != compID)
                boost::add_edge(compID, mergeNeigh, levelGraph);

            Log::trace("Level {0}: merge {1} to {2}", currentLevel, compID, mergeNeigh);

            progress.update();
        }
        progress.finish();
    }

    void ImageHierarchy::connectMostSimilarComponents(uint64_t numComponentsCurrent, uint64_t currentLevel, utils::BoostGraph& levelGraph)
    {
        uint64_t& zeroSimCounter = _stats.zeroSimilarityCount.emplace_back(0);
        uint64_t& forcedMergeCounter = _stats.forcedMergeCount.emplace_back(0);

        utils::Graph distances = computeDistances(numComponentsCurrent, currentLevel);

        float thresh = std::numeric_limits<float>::max();

        if (_ihs.maxDist > 0.f)
        {
            thresh = _ihs.maxDist;

            if (_ihs.usePercentile)
            {
                auto quantile = utils::computeQuantile(distances.getKnnDistances(), _ihs.maxDist, { 0.f, -1.f, std::numeric_limits<float>::max() });

                if (quantile < 0)
                {
                    Log::warn("ImageHierarchy::computeBoruvkaHierarchy: findPercentile could not find percentile, reset thresh to default");
                    quantile = std::numeric_limits<float>::max();
                }

                const auto maxDistance = std::max_element(SPH_PARALLEL_EXECUTION
                    distances.getKnnDistances().begin(), 
                    distances.getKnnDistances().end());

                Log::info("ImageHierarchy::computeBoruvkaHierarchy: percentile {0} yields threshold {1} (of max distance {2})", _ihs.maxDist, quantile, *maxDistance);
                thresh = static_cast<float>(quantile);
            }
        }

        assert(thresh >= 0);

        if (_ihs.mergeMultiple)
            mergeAllBelow(thresh, numComponentsCurrent, currentLevel, distances, levelGraph, zeroSimCounter, forcedMergeCounter);    // This is strictly speaking NOT how Boruvka works!
        else
            mergeMinBelow(thresh, numComponentsCurrent, currentLevel, distances, levelGraph, zeroSimCounter, forcedMergeCounter);

        Log::info("ImageHierarchy::computeBoruvkaHierarchy: {} components with no similarity on current level {} ({:.2f}%)", zeroSimCounter, currentLevel, 100.f * static_cast<float>(zeroSimCounter) / numComponentsCurrent);
        
        if (_ihs.isAlwaysMerge())
            Log::info("ImageHierarchy::computeBoruvkaHierarchy: Forced to merge with a random neighbor: {}", forcedMergeCounter);
    }

    void ImageHierarchy::computeBoruvkaHierarchy()
    {
        auto numTrees = _data.getNumPoints();

        Log::info("ImageHierarchy::computeBoruvkaHierarchy: {0} trees on level 0 (pixel level)", numTrees);

        // stop when...
        //  - there is no more merging
        //  - the last to merging rates where lower than one percent
        auto reductionStagnates = [&]() -> bool {
            if (_stats.reductionRates.back() == 100.f ||
                (_stats.reductionRates.size() > 2 && (_stats.reductionRates[_stats.reductionRates.size() - 1] > _ihs.minReduction && _stats.reductionRates[_stats.reductionRates.size() - 2] > _ihs.minReduction)))
                return true;

            return false;
            };

        auto setupSimHelper = [this]() {
            if (_ihs.componentSim == utils::ComponentSim::GEO_CENTROID || _ihs.componentSim == utils::ComponentSim::GEO_WALKS)
            {
                Log::info("ImageHierarchy::computeBoruvkaHierarchy: setup boost graph");
                _hierarchy.bgraph = std::make_unique<utils::BoostGraph>(utils::createBoostGraph(_dataKnnGraph));
            }
            };

        auto cleanupSimHelper = [this]()
            {
                if (_ihs.componentSim == utils::ComponentSim::GEO_CENTROID || _ihs.componentSim == utils::ComponentSim::GEO_WALKS)
                {
                    //Log::info("ImageHierarchy::computeBoruvkaHierarchy: cleanup boost graph");
                    // nothing to do here
                }
            };

        assert(_ihs.minNumComp > 0);

        setupSimHelper();

        for (uint64_t currentLevel = 0; numTrees > _ihs.minNumComp; currentLevel++)
        {
            if (_ihs.maxLevels >= 0 && currentLevel >= static_cast<uint64_t>(_ihs.maxLevels))
            {
                Log::info("ImageHierarchy::computeBoruvkaHierarchy: reached max level {}. Stopping hierarchy computation...", currentLevel);
                break;
            }

            Log::info("ImageHierarchy::computeBoruvkaHierarchy: merging on current level {} starts computing...", currentLevel);

            const uint64_t numComponentsCurrent = _hierarchy.numComponents[currentLevel];

            // STEP 0: add nodes (numComponentsCurrent) to helper graph, connections come in next step
            utils::BoostGraph levelGraph = {};
            for ([[maybe_unused]]const auto& compID : std::views::iota(0ull, numComponentsCurrent))
                [[maybe_unused]] auto vertex_descriptor = boost::add_vertex(levelGraph);

            // STEP 1: find most similar spatially neighboring component of each component
            //         each component will create one edge to the component that it wants to merge with
            connectMostSimilarComponents(numComponentsCurrent, currentLevel, levelGraph);

            // STEP 2: find connected components, these will be the new components on the next hierarchy level
            vi64 componentLabelsNext = {};
            const int64_t numComponentsNext = utils::labelGraphWeakComponents(levelGraph, componentLabelsNext);
            assert(numComponentsCurrent == componentLabelsNext.size());

#if SPH_DEBUG
            // Checks if always merging is a thing (_minSim == -1.f)
            if (_ihs.isAlwaysMerge())
            {
                // Each ID has to occur at least two times
                std::vector<int64_t> countConnectedComps(componentLabelsNext.size());
                for (const auto& label : componentLabelsNext)
                    countConnectedComps[label]++;
                for (const auto& countConnectedComp : countConnectedComps)
                    assert(countConnectedComp > 1);

                // Every level must have less than half the number of components
                assert(static_cast<uint64_t>(numComponentsNext) <= _hierarchy.numComponents.back() / 2);
            }
#endif // SPH_DEBUG

            _stats.reductionRates.push_back(100.f * static_cast<float>(numComponentsNext) / _hierarchy.numComponents.back());
            assert(_stats.reductionRates.back() <= 100.f);

            Log::info("ImageHierarchy::computeBoruvkaHierarchy: {} trees on next level {} (reduction to {:.2f}%)", numComponentsNext, currentLevel + 1, _stats.reductionRates.back());

            if (reductionStagnates())
            {
                Log::info("ImageHierarchy::computeBoruvkaHierarchy: no significant reduction - level is not added. Stopping hierarchy computation...");
                break;
            }

            // STEP 3: update hierarchy
            //         store parent and child relations as well as new spatial neighbors
            Log::info("ImageHierarchy::computeBoruvkaHierarchy: update hierarchy helper");

            // Adaptive random walk length: reduce length according to reduction rate
            if ((_ihs.componentSim == utils::ComponentSim::NEIGH_WALKS ||
                 _ihs.componentSim == utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP ||
                 _ihs.componentSim == utils::ComponentSim::GEO_WALKS) &&
                _ihs.rwHandling != utils::RandomWalkHandling::MERGE_RW_ONLY
                )
            {
                const uint64_t currentRandomWalkLength = _ihs.rwRandomWalkLengths.back();
                float reductionRate = 0.f;

                if (_ihs.rwReduction == utils::RandomWalkReduction::NONE) {
                    reductionRate = 1.f;
                }
                else if (_ihs.rwReduction == utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION) {
                    reductionRate = _stats.reductionRates.back() / 100.f;
                }
                else if (_ihs.rwReduction == utils::RandomWalkReduction::PROPORTIONAL_DOUBLE) {
                    reductionRate = _stats.reductionRates.back() / 100.f * 2.f;
                }
                else if (_ihs.rwReduction == utils::RandomWalkReduction::PROPORTIONAL_HALF) {
                    reductionRate = _stats.reductionRates.back() / 100.f * 0.5f;
                }
                else if (_ihs.rwReduction == utils::RandomWalkReduction::CONSTANT) {
                    reductionRate = 0.5f;
                }
                else if (_ihs.rwReduction == utils::RandomWalkReduction::CONSTANT_LOW) {
                    reductionRate = 0.75f;
                }
                else if (_ihs.rwReduction == utils::RandomWalkReduction::CONSTANT_HIGH) {
                    reductionRate = 0.25f;
                }
                else {
                    reductionRate = 1.f;
                    Log::warn("ImageHierarchy::computeBoruvkaHierarchy: random walk length reduction RandomWalkReduction {0} not implemented, defaulting to NONE", _ihs.rwReduction);
                }

                reductionRate = std::clamp(reductionRate, 0.f, 1.f);

                uint64_t nextRandomWalkLength = static_cast<uint64_t>(reductionRate * currentRandomWalkLength);
                nextRandomWalkLength = std::clamp(nextRandomWalkLength, _rws.minimumSingleWalkLength, _ihs.rwRandomWalkLengths.front());
                _ihs.rwRandomWalkLengths.push_back(nextRandomWalkLength);

                Log::info("ImageHierarchy::computeBoruvkaHierarchy: random walk length reduction rate of {0}", reductionRate);
                Log::info("ImageHierarchy::computeBoruvkaHierarchy: reduce random walk length to {0} from {1} (using {2})", nextRandomWalkLength, currentRandomWalkLength, _ihs.rwReduction);
            }

            // Add next level
            auto nxtHierarchyInfo = utils::Hierarchy::AddLevelInfo{ _rws, numComponentsNext, componentLabelsNext };

            if(!_ihs.rwRandomWalkLengths.empty())
                nxtHierarchyInfo.rwsSettings.singleWalkLength = _ihs.rwRandomWalkLengths.back();
    
            _hierarchy.addLevel(nxtHierarchyInfo);

            // STEP 4: stats
            //         Log how many components were not merged
            uint64_t numNotMergedComponents = _hierarchy.notMergedNodes.back().size();
            _stats.notMergedComponents.push_back(numNotMergedComponents);

            if(_hierarchy.randomWalks.size() > 1)
            {
                utils::SparseMatrixStats rwStats = utils::sparseMatrixStats(_hierarchy.randomWalks.back());
                _stats.rwSparsities.push_back(rwStats.sparsityEffective);
            }

            if (_hierarchy.mergedDataGraphs.size() > 0)
            {
                auto mergedDataGraph = _hierarchy.mergedDataGraphs.back().get();
                utils::SparseMatrixStats mergedDataStats = utils::sparseMatrixStats(mergedDataGraph);
                _stats.mergedDataSparsities.push_back(mergedDataStats.sparsityEffective);
            }

            Log::info("ImageHierarchy::computeBoruvkaHierarchy: {} components that were not merged on current level {} ({:.2f}%)", numNotMergedComponents, currentLevel, 100.f * static_cast<float>(numNotMergedComponents) / numComponentsCurrent);

            // update loop condition
            numTrees = numComponentsNext;

            if(numTrees <= _ihs.minNumComp)
                Log::info("ImageHierarchy::computeBoruvkaHierarchy: new number of components ({}) is smaller than min number of components ({}). Stopping hierarchy computation...", numTrees, _ihs.minNumComp);

        }

        _stats.numComponents = _hierarchy.numComponents;

        cleanupSimHelper();

        Log::info("ImageHierarchy::computeBoruvkaHierarchy: finished component hierarchy computation with {0} levels (including the data level)", _hierarchy.getNumLevels());
    }


    std::vector<uint64_t> ImageHierarchy::getSpatialNeighbors(const utils::ComponentID& cid) const
    {
        std::vector<uint64_t> spNeigh;

        // If we are on the lowest level, the id only represents itself
        if (cid.level == 0)
            utils::pixelNeighborIDs(_numCols, _numRows, _neighConnection, cid.id, spNeigh);
        else
            spNeigh = _hierarchy.spatialNeighbors[cid.level - 1][cid.id];

        return spNeigh;
    }

    bool ImageHierarchy::writeStats(const std::string& fileName) const {

        Log::info("ImageHierarchy::writeStats: write to {}", fileName);

        // store parameters in json file
        nlohmann::json parameters;

        parameters["zeroSimilarityCount"]   = _stats.zeroSimilarityCount;
        parameters["forcedMergeCount"]      = _stats.forcedMergeCount;
        parameters["reductionRates"]        = _stats.reductionRates;
        parameters["rwSparsities"]          = _stats.rwSparsities;
        parameters["mergedDataSparsities"]  = _stats.mergedDataSparsities;
        parameters["numComponents"]         = _stats.numComponents;
        parameters["notMergedComponents"]   = _stats.notMergedComponents;
        parameters["numLevels"]             = _stats.numComponents.size();

        // Write to file
        bool successWritingParameters = utils::writeJsonToDisk(fileName, parameters);

        if (!successWritingParameters)
            return false;

        return true;
    }

    bool ImageHierarchy::checkCacheParameters(const std::string& fileName) const {
        auto [parameters, successLoadingParameters] = utils::loadJsonFromDisk(fileName);

        if (!successLoadingParameters)
            return false;

        if (!isVersionCompatible(parameters))
            return false;

        if (!utils::checkEntry("Input data name", parameters, _cacheFileName)) return false;
        if (!utils::checkEntry("Number columns", parameters, _numCols)) return false;
        if (!utils::checkEntry("Number rows", parameters, _numRows)) return false;

        if(!utils::checkSettings(parameters, _ihs)) return false;
        if(!utils::checkSettings(parameters, _rws)) return false;

        Log::info("Parameters of cache correspond to current settings.");

        return true;
    }

    bool ImageHierarchy::writeCacheParameters(const std::string& fileName) const
    {
        Log::info("ImageHierarchy::writeCacheParameters: write to " + fileName);

        // store parameters in json file
        nlohmann::json parameters;
        parameters["## VERSION ##"] = _cacheParameterVersion;

        parameters["Input data name"] = _cacheFileName;
        parameters["Number columns"] = _numCols;
        parameters["Number rows"] = _numRows;

        utils::addToJson(_ihs, parameters);
        utils::addToJson(_rws, parameters);

        // Write to file
        bool successWritingParameters = utils::writeJsonToDisk(fileName, parameters);

        if (!successWritingParameters)
            return false;

        return true;
    }

    bool ImageHierarchy::loadCacheHierarchy(const std::string& fileNameBase)
    {
        _hierarchy.clear();

        bool success = false;
        std::string fileName = "";

        auto checkLoad = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Loading failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "NumComponents.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecFromBinary(fileName, _hierarchy.numComponents);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "Parents.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecOfVecFromBinary(fileName, _hierarchy.parents);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "Children.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecOfVecOfVecFromBinary(fileName, _hierarchy.children);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "SpatialNeighbors.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecOfVecOfVecFromBinary(fileName, _hierarchy.spatialNeighbors);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "PixelComponents.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecOfVecFromBinary(fileName, _hierarchy.pixelComponents);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "MapFromLevelToBottom.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecOfVecOfVecFromBinary(fileName, _hierarchy.mapFromLevelToPixel);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "RandomWalkSimilarities.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecsOfSparseMatSPHFromBinary(fileName, _hierarchy.randomWalks);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "NotMergedNodes.cache";
        Log::info("ImageHierarchy::loadCacheHierarchy: loading " + fileName);
        success = utils::loadCompressedVecOfVecFromBinary(fileName, _hierarchy.notMergedNodes);
        if (!checkLoad(success, fileName)) return success;

        {
            fileName = fileNameBase + "mergedDataGraphsNum.cache";

            Log::info("ImageHierarchy::loadCacheHierarchy: writing {0}", fileName);
            std::vector<uint64_t> mergedDataGraphsNum;
            success = utils::loadVecFromBinary(fileName, mergedDataGraphsNum);
            if (!checkLoad(success, fileName)) return success;

            if(mergedDataGraphsNum.size() == 1 && mergedDataGraphsNum[0] > 1)
            {
                _hierarchy.mergedDataGraphs.push_back(std::make_unique<utils::GraphView>(_dataKnnGraph));

                for (size_t num = 1; num < mergedDataGraphsNum[0]; num++)
                {
                    fileName = fileNameBase + "mergedDataGraphsNum.cache_" + std::to_string(num);
                    Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);

                    utils::Graph currentGraph;
                    success = utils::loadCompressedGraphFromBinary(fileName, currentGraph);
                    if (!checkLoad(success, fileName)) return success;

                    _hierarchy.mergedDataGraphs.emplace_back(std::make_unique<utils::Graph>(std::move(currentGraph)));
                }
            }
        }

        return success;
    }

    bool ImageHierarchy::writeCacheHierarchy(const std::string& fileNameBase) const
    {
        bool success = false;
        std::string fileName = "";

        auto checkWrite = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Writing failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "NumComponents.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecToBinary(fileName, _hierarchy.numComponents);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "Parents.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecOfVecToBinary(fileName, _hierarchy.parents);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "Children.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecOfVecOfVecToBinary(fileName, _hierarchy.children);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "SpatialNeighbors.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecOfVecOfVecToBinary(fileName, _hierarchy.spatialNeighbors);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "PixelComponents.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecOfVecToBinary(fileName, _hierarchy.pixelComponents);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "MapFromLevelToBottom.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecOfVecOfVecToBinary(fileName, _hierarchy.mapFromLevelToPixel);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "RandomWalkSimilarities.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecsOfSparseMatSPHToBinary(fileName, _hierarchy.randomWalks);
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "NotMergedNodes.cache";
        Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);
        success = utils::writeCompressedVecOfVecToBinary(fileName, _hierarchy.notMergedNodes);
        if (!checkWrite(success, fileName)) return success;

        {
            fileName = fileNameBase + "mergedDataGraphsNum.cache";

            Log::info("ImageHierarchy::writeCacheHierarchy: writing {0}", fileName);
            std::vector<uint64_t> mergedDataGraphsNum = { _hierarchy.mergedDataGraphs.size() };
            success = utils::writeVecToBinary(fileName, mergedDataGraphsNum);
            if (!checkWrite(success, fileName)) return success;

            for (size_t num = 1; num < _hierarchy.mergedDataGraphs.size(); num++)
            {
                fileName = fileNameBase + "mergedDataGraphsNum.cache_" + std::to_string(num);
                Log::info("ImageHierarchy::writeCacheHierarchy: writing " + fileName);

                const utils::GraphInterface* currentInterface = _hierarchy.mergedDataGraphs[num].get();
                const utils::Graph* currentGraph = dynamic_cast<const utils::Graph*>(currentInterface);

                if(currentGraph)
                    success = utils::writeCompressedGraphToBinary(fileName, currentGraph->getGraphView());

                if (!checkWrite(success, fileName)) return success;

                if (!checkWrite(success, fileName)) return success;
            }
        }
        
        // TODO: missing -> _hierarchy.settings 
        // TODO: missing -> _hierarchy.bgraph 

        return success;
    }


    bool ImageHierarchy::loadCacheImageHierarchy(const std::string& fileNameBase)
    {
        _dataLevelProbdist.clear();

        bool success = false;
        std::string fileName = "";

        auto checkLoad = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Loading failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "DataLevelProbdist.cache";
        Log::info("ImageHierarchy::loadCacheImageHierarchy: loading " + fileName);
        success = utils::loadCompressedSparseMatSPHFromBinary(fileName, _dataLevelProbdist);
        if (!checkLoad(success, fileName)) return success;

        return success;
    }

    bool ImageHierarchy::writeCacheImageHierarchy(const std::string& fileNameBase) const
    {
        bool success = false;
        std::string fileName = "";

        auto checkWrite = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Writing failed: file {0}", f); return false; }
            return true;
            };

        // in case only ImageHierarchy was run
        // we need this later in LevelSimilarities
        // TODO: no need to write it to disk a second time there
        fileName = fileNameBase + "DataLevelProbdist.cache";
        Log::info("ImageHierarchy::writeCacheImageHierarchy: writing " + fileName);
        success = utils::writeCompressedSparseMatSPHToBinary(fileName, _dataLevelProbdist);
        if (!checkWrite(success, fileName)) return success;

        return success;
    }

    bool ImageHierarchy::loadCache()
    {
        if (!getCachingActive())
        {
            Log::info("ImageHierarchy::loadCache: Caching is not active. Use setCachingActive(true) if desired.");
            return cachingFailure();
        }

        if (!cacheDependencyIsValid())
        {
            Log::info("ImageHierarchy::loadCache: Cannot load cache since dependency could not load cache either.");
            return cachingFailure();
        }

        Log::info("ImageHierarchy::loadCache: attempt to load cache from " + _cachePath.string());

        auto fullPath = (_cachePath / _cacheFileName).string() + "_ihs";
        auto pathParameter = fullPath + "_parametersHierarchy.cache";
        auto pathHierarchy = fullPath + "_hierarchy";
        auto pathImageHierarchy = fullPath + "_imageHierarchy";

        std::vector<Path> pathList = { pathParameter };

        pathList.emplace_back(pathHierarchy + "NumComponents" + ".cache");
        pathList.emplace_back(pathHierarchy + "Parents" + ".cache");
        pathList.emplace_back(pathHierarchy + "Children" + ".cache");
        pathList.emplace_back(pathHierarchy + "SpatialNeighbors" + ".cache");
        pathList.emplace_back(pathHierarchy + "PixelComponents" + ".cache");
        pathList.emplace_back(pathHierarchy + "MapFromLevelToBottom" + ".cache");
        pathList.emplace_back(pathHierarchy + "RandomWalkSimilarities" + ".cache_0");   // special case: large files are split, always at least one file though
 
        // TODO: missing (not saved) -> _hierarchy.settings 
        // TODO: missing (not saved) -> _hierarchy.bgraph 

        for (const Path& path : pathList)
        {
            if (!(std::filesystem::exists(path)))
            {
                Log::info("Loading cache failed: No file exists at: " + path.string());
                return cachingFailure();
            }
        }

        if (!checkCacheParameters(pathParameter))
        {
            Log::warn("Loading cache failed: Current settings are different from cached parameters.");
            return cachingFailure();
        }

        if (!loadCacheHierarchy(pathHierarchy))
        {
            Log::warn("Loading cache failed: " + pathHierarchy);
            return cachingFailure();
        }

        if (!loadCacheImageHierarchy(pathImageHierarchy))
        {
            Log::warn("Loading cache failed: " + pathImageHierarchy);
            return cachingFailure();
        }

        // Log some stats
        Log::info("Loaded a hierarchy with {0} levels", _hierarchy.numComponents.size());
        Log::info("Level 0 with {} components", _hierarchy.numComponents[0]);

        for (uint64_t i = 1; i < _hierarchy.numComponents.size(); i++)
            Log::info("Level {} with {} components (reduction to {:.2f}%)", i, _hierarchy.numComponents[i], 100.f * static_cast<float>(_hierarchy.numComponents[i]) / _hierarchy.numComponents[i - 1]);

        Log::info("ImageHierarchy::loadCache: finished");
        return cachingSuccess();
    }

    bool ImageHierarchy::writeCache() const
    {
        if (!mayCache("ImageHierarchy"))
            return false;

        Log::info("ImageHierarchy::writeCache: save cache to " + _cachePath.string());

        auto fullPath = (_cachePath / _cacheFileName).string() + "_ihs";
        auto pathParameter = fullPath + "_parametersHierarchy.cache";
        auto pathHierarchy = fullPath + "_hierarchy";
        auto pathImageHierarchy = fullPath + "_imageHierarchy";

        bool successParam = writeCacheParameters(pathParameter);
        bool successHierarchy = writeCacheHierarchy(pathHierarchy);
        bool successImageHierarchy = writeCacheImageHierarchy(pathImageHierarchy);

        Log::info("ImageHierarchy::writeCache: finished");
        return successParam && successHierarchy && successImageHierarchy;
    }

} // namespace sph
