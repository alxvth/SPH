#include "RunEvaluation.hpp"

#include "EvaluationSettings.hpp"

#include <sph/ComputeEmbedding.hpp>
#include <sph/ComputeHierarchy.hpp>
#include <sph/ImageHierarchy.hpp>
#include <sph/LevelSimilarities.hpp>
#include <sph/NearestNeighbors.hpp>

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/Embedding.hpp>
#include <sph/utils/EvalIO.hpp>
#include <sph/utils/FileIO.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/HDILibHelper.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/Math.hpp>
#include <sph/utils/Scaler.hpp>
#include <sph/utils/Settings.hpp>
#include <sph/utils/Statistics.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <variant>

#include <range/v3/view/cartesian_product.hpp>

namespace sph::eval {

    namespace fs = std::filesystem;

void runEvaluation(const EvalSettings& settings) {

    const std::string currentTime       = utils::getCurrentDateTimeNumbers();
    const std::string currentTimeHash   = utils::createShortHash(currentTime);
    utils::Logger::setLogPath(settings.cachePathBase / fmt::format("Log_{0}.txt", currentTime));
    Log::info("Starting evaluation: {}", utils::getCurrentDateTimeHuman());

    // Check setting validity
    utils::ensurePathExists(settings.cachePathBase);

    // Init setting structs
    auto nns    = NearestNeighborsSettings();
    auto ihs    = ImageHierarchySettings();
    auto lss    = LevelSimilaritiesSettings();
    auto rws    = utils::RandomWalkSettings();
    auto ces    = ComputeEmbeddingSettings();
    auto cs     = CacheSettings();
    auto cs_knn = CacheSettings();
    auto cs_cc  = CacheSettings();
    auto cs_geo = CacheSettings();

    // Define static settings

    nns.computeConnectComponents    = true;                                 // for logging, we always want to compute the CC even if we do not use them

    ihs.mergeMultiple               = false;
    ihs.usePercentile               = false;
    ihs.maxDist                     = 0;
    ihs.minNumComp                  = 1;                                    // eval default: 1
    ihs.minReduction                = 98.0f;                                // 99.5 seems to high for images where noise is present
    ihs.numGeodesicSamples          = 100;                                  // not used for settings.componentSim == utils::ComponentSim::NEIGH_WALKS, std::numeric_limits<size_t>::max() would create large full distance matrices on higher levels
    ihs.maxLevels                   = 10;                                   // eval default: 10
    ihs.verbose                     = false;                                // This will print A LOT

    rws.pruneSteps                  = 0;                                    // 

    lss.forceComputeDistances       = false;                                // maybe interesting for debugging

    cs.cacheActive                  = false;
    fs::path externalCachePath      = {};
    cs_knn.cacheActive              = true;
    cs_knn.ignoreSubfolder          = true;
    cs_cc.cacheActive               = true;
    cs_geo.cacheActive              = true;

    ces.tsne.perplexity             = -1;                                   // will be populated automatically based on number of neighbors on level
    ces.tsne.symmetricProbDist      = true;                                 // we compute symmetric probability distribution here
    ces.umap.singleStep             = false;

    // Configure eval steps
    const bool skipLevelSimilarities      = settings.skipLevelSimilarities;
    const bool skipEmbeddingTSNE          = settings.skipEmbeddingTSNE;
    const bool skipEmbeddingUMAP          = settings.skipEmbeddingUMAP;
    const bool skipExistingSettings       = settings.skipExistingSettings;
    const bool skipSaveStructuresToDisk   = settings.skipSaveStructuresToDisk;
    const bool initLevelEmbWithPrevious   = settings.initLevelEmbWithPrevious;

    // Setup tests
    const auto settingsCombinationsGeneral = ranges::views::cartesian_product(
        /*  0 */ settings.imageNames,
        /*  1 */ settings.componentSim,
        /*  2 */ settings.dataInputNorm,
        /*  3 */ settings.dataDistNorm,
        /*  4 */ settings.neighborConnections,
        /*  5 */ settings.neighborSymmetries,
        /*  6 */ settings.neighborConnectComponents,
        /*  7 */ settings.nKnns,
        /*  8 */ settings.knnMetrics,
        /*  9 */ settings.weightTransitionBySize
        );

    const auto settingsCombinationsRandomWalks = ranges::views::cartesian_product(
        /* 0 */ settings.randomWalkNums,
        /* 1 */ settings.randomWalkLens,
        /* 2 */ settings.randomWalkPairSims,
        /* 3 */ settings.randomWalkStepWeight,
        /* 4 */ settings.randomWalkWeightSize,
        /* 5 */ settings.randomWalkHandling,
        /* 6 */ settings.randomWalkReduction
    );

    // For geodesic settings, we do not need to iterate over random walks settings and fall back to this
    const std::vector<uint64_t> randomWalkNumsGeo = { 0 };
    const std::vector<uint64_t> randomWalkLensGeo = { 0 };
    const std::vector<bool> randomWalkPairSimsGeo = { false };
    const std::vector<utils::ImportanceWeighting> randomWalkStepWeightGeo = { utils::ImportanceWeighting::NORMAL };
    const std::vector<bool> randomWalkWeightSizeGeo = { false };
    const std::vector<utils::RandomWalkHandling> randomWalkHandlingGeo = { utils::RandomWalkHandling::MERGE_RW_ONLY };
    const std::vector<utils::RandomWalkReduction> randomWalkReductionGeo = { utils::RandomWalkReduction::NONE };

    const auto settingsCombinationsRandomWalksDefault = ranges::views::cartesian_product(
        /* 0 */ randomWalkNumsGeo,
        /* 1 */ randomWalkLensGeo,
        /* 2 */ randomWalkPairSimsGeo,
        /* 3 */ randomWalkStepWeightGeo,
        /* 4 */ randomWalkWeightSizeGeo,
        /* 5 */ randomWalkHandlingGeo,
        /* 6 */ randomWalkReductionGeo
    );

    const size_t combinationsGeneral = settingsCombinationsGeneral.size();
    const size_t combinationsRandom = settingsCombinationsRandomWalks.size();
    const size_t generalSettingsPerSimilarity = combinationsGeneral / settings.componentSim.size();
    const size_t numRandomWalkSettings = numRandomWalkSettingComponentSim(settings.componentSim);
    const size_t numGeodesicSettings = settings.componentSim.size() - numRandomWalkSettings;
    const size_t numRuns = generalSettingsPerSimilarity * numGeodesicSettings + generalSettingsPerSimilarity * combinationsRandom * numRandomWalkSettings;

    size_t runID = 0;
    std::vector<std::pair<std::string, std::string>> settingHashes = {};

    for (const auto& settingsGeneral : settingsCombinationsGeneral)
    {
        // general settings
        const std::string imageFileName = static_cast<std::string>(std::get<0>(settingsGeneral));
        const utils::ComponentSim componentSim = static_cast<utils::ComponentSim>(std::get<1>(settingsGeneral));
        const utils::Scaler dataNormalization = static_cast<utils::Scaler>(std::get<2>(settingsGeneral));

        // knn settings 
        const utils::NormalizationScheme normScheme = static_cast<utils::NormalizationScheme>(std::get<3>(settingsGeneral));
        const utils::NeighConnection neighborConnection = static_cast<utils::NeighConnection>(std::get<4>(settingsGeneral));
        const bool neighborSymmetric = static_cast<bool>(std::get<5>(settingsGeneral));
        const bool neighborConnectComponents = static_cast<bool>(std::get<6>(settingsGeneral));
        const uint64_t nKnns = static_cast<uint64_t>(std::get<7>(settingsGeneral));
        const utils::KnnMetric nearestNeighborMetric = static_cast<utils::KnnMetric>(std::get<8>(settingsGeneral));

        const EvalSettingNeighbors evalSettingNeighbors = { neighborConnection, neighborSymmetric, neighborConnectComponents, nKnns, nearestNeighborMetric };

        // embedding settings
        const bool weightTransitionBySize = static_cast<bool>(std::get<9>(settingsGeneral));

        // settings flow control
        const bool settingHasRandomWalks = !onlyGeodesicSettings(componentSim);
        const auto* settingsCombinationsRandomWalksCurrent = settingHasRandomWalks ? &settingsCombinationsRandomWalks : &settingsCombinationsRandomWalksDefault;
        
        for (const auto& settingsRandomWalks : *settingsCombinationsRandomWalksCurrent)
        {
            // random walk settings
            const uint64_t randomWalkNum = static_cast<uint64_t>(std::get<0>(settingsRandomWalks));
            const uint64_t randomWalkLen = static_cast<uint64_t>(std::get<1>(settingsRandomWalks));
            const bool randomWalkPairSims = static_cast<bool>(std::get<2>(settingsRandomWalks));
            const utils::ImportanceWeighting randomWalkWeight = static_cast<utils::ImportanceWeighting>(std::get<3>(settingsRandomWalks));
            const bool randomWalkWeightBySize = static_cast<bool>(std::get<4>(settingsRandomWalks));
            const utils::RandomWalkHandling randomWalkHandling = static_cast<utils::RandomWalkHandling>(std::get<5>(settingsRandomWalks));
            const utils::RandomWalkReduction randomWalkReduction = static_cast<utils::RandomWalkReduction>(std::get<6>(settingsRandomWalks));

            const EvalSettingRandomWalks evalSettingRandomWalks = { randomWalkNum, randomWalkLen, randomWalkWeight, randomWalkWeightBySize, randomWalkPairSims, randomWalkHandling, randomWalkReduction };

            runID++;

            Log::info("### ####### ###");
            Log::info("### NEW RUN ###");
            Log::info("### ####### ###");
            Log::info("Run {0} of {1}", runID, numRuns);

            constexpr uint64_t memLim = 200ll * 500 * 200;
            if (randomWalkNum * randomWalkLen * nKnns > memLim)
            {
                Log::info("Setting combination of rwNum {0}, rwLen {1} and nbKnn {2} exceeds limit of {3}", randomWalkNum, randomWalkLen, nKnns, memLim);
                continue;
            }

            if (dataNormalization != utils::Scaler::NONE && nearestNeighborMetric == utils::KnnMetric::COSINE)
            {
                Log::info("Setting combination of dataNormalization TRUE and nearestNeighborMetric COSINE probably does not make sense, skipping...");
                continue;
            }

            const fs::path imageNameWoExtension     = fs::path(imageFileName).stem();
            const fs::path currentSavePathBase      = settings.saveSubFolder.empty() ? settings.cachePathBase / imageNameWoExtension : settings.cachePathBase / settings.saveSubFolder;
            const std::string saveSettings          = getSettingsString(componentSim, normScheme, dataNormalization, weightTransitionBySize, evalSettingNeighbors, evalSettingRandomWalks);
            const std::string saveSettingsHash      = utils::createShortHash(saveSettings);
            const std::string saveIndicator         = settingHasRandomWalks ? fmt::format("{0}_{1}", componentSim, randomWalkHandling) : fmt::format("{0}", componentSim);
            const std::string currentSavePathFolder = fmt::format("{0}_{1}_{2}_{3}", saveIndicator, saveSettingsHash, currentTime, runID);
            const fs::path currentSavePath          = currentSavePathBase / currentSavePathFolder;

            utils::ensurePathExists(currentSavePathBase);
            utils::ensurePathExists(currentSavePath);

            Log::info("Saving to: {0}", currentSavePath.string());

            if (skipExistingSettings && !utils::folderIsEmpty(currentSavePath))
            {
                Log::info("Save folder is not empty; skipping this setting.");
                continue;
            }

            // load current data
            utils::ImageStack img;
            const fs::path imagePath = settings.inputPath / imageFileName;

            Log::info("Load data from: {0}", imagePath.string());
            if (settings.imgFormat == ImageFormat::TIFFSTACK)
                img = utils::loadTiffImageStack(imagePath);
            else
                img = utils::loadRGBdata(imagePath);

            if (img.data.getNumPoints() == 0)
            {
                Log::warn("Image has 0 points, skipping this one...");
                continue;
            }

            // assign current settings
            uint64_t nbNnData = nKnns;
            if (nbNnData == 0)
            {
                float perplexity = img.data.getNumPoints() / 100.f;
                perplexity = std::clamp(perplexity, 10.f, 100.f);
                nbNnData = static_cast<int64_t>(perplexity) * 3;   // 3 is perplexity multiplier
            }
            nbNnData++; // point itself will be one of the computed nn

            rws.numRandomWalks = randomWalkNum;
            rws.singleWalkLength = randomWalkLen;
            rws.importanceWeighting = randomWalkWeight;

            ihs.componentSim = componentSim;
            ihs.neighborConnection = neighborConnection;
            ihs.rwHandling = randomWalkHandling;
            ihs.rwWeightMergeBySize = randomWalkWeightBySize;
            ihs.rwReduction = randomWalkReduction;
            ihs.normKnnDistances = normScheme;

            nns.numNearestNeighbors = nbNnData;
            nns.symmetricNeighbors = neighborSymmetric;
            nns.neighborConnectComponents = neighborConnectComponents;
            nns.knnMetric = nearestNeighborMetric;

            nns.knnIndex = NearestNeighbors::indexHeuristic(img.data.numPoints);

            lss.ks = { static_cast<int64_t>(nbNnData) };
            lss.componentSim = componentSim;
            lss.randomWalkPairSims = randomWalkPairSims;
            lss.weightTransitionBySize = weightTransitionBySize;
            lss.normalizeProbDist = normScheme;

            if (externalCachePath.empty())
                lss.computeSymmetricProbDist = utils::NormalizationScheme::NONE;     // we compute symmetric probDist for both UMAP and t-SNE later

            const std::string cc_suffix_knn = std::to_string(nbNnData) + "_" + std::to_string(utils::to_underlying(nns.knnMetric)) + "_" + std::to_string(utils::to_underlying(dataNormalization));
            const std::string cc_suffix_cc = cc_suffix_knn + "_" + std::to_string(neighborSymmetric);

            cs.fileName     = imageFileName;
            cs_knn.fileName = imageNameWoExtension.string();
            cs_cc.fileName  = imageFileName;
            cs_geo.fileName = imageFileName;

            if (externalCachePath.empty()) {
                cs.path     = currentSavePath.string();
                cs_knn.path = (currentSavePathBase / "knn" / cc_suffix_knn).string();
                cs_cc.path  = (currentSavePathBase / "wcc" / cc_suffix_cc).string();
                cs_geo.path = (currentSavePathBase / "geo" / cc_suffix_cc).string();
            }
            else {
                cs.path     = externalCachePath.string();
                cs_knn.path = (externalCachePath / "sph-cache").string();
                cs_cc.path  = cs_knn.path;
                cs_geo.path = cs_knn.path;
            }

            if (cs_knn.cacheActive)
                utils::ensurePathExists(cs_knn.path);

            if (cs_cc.cacheActive)
                utils::ensurePathExists(cs_cc.path);

            if (cs_geo.cacheActive)
                utils::ensurePathExists(cs_geo.path);

            // log settings
            Log::info("File {0} with size {1} x {2} (= {3}) and {4} dimensions", imageFileName, img.height, img.width, img.data.numPoints, img.data.numDimensions);
            Log::info("component similarity: {0}", ihs.componentSim);
            Log::info("nbCon: {0}, nbKnn: {1}", ihs.neighborConnection, nns.numNearestNeighbors);
            Log::info("nbSym: {0}, nbComp {1}", nns.symmetricNeighbors, nns.neighborConnectComponents);
            Log::info("ddNor: {0}, wTrbS: {1}", lss.normalizeProbDist, lss.weightTransitionBySize);

            if (settingHasRandomWalks)
            {
                Log::info("rwNum: {0}, rwLen: {1}", rws.numRandomWalks, rws.singleWalkLength);
                Log::info("rwWei: {0}, rwSim: {1}", rws.importanceWeighting, lss.componentSim);
                Log::info("rwHan: {0}, rwSiz: {1}", ihs.rwHandling, ihs.rwWeightMergeBySize);
                Log::info("rwRed: {0}", ihs.rwReduction);
                Log::info("rwPai: {0} (during level sim)", lss.randomWalkPairSims);
            }

            // Copy the settings file
            const fs::path outSettingsPath = currentSavePathBase / (fmt::format("{0}_settings{1}", currentTime, settings.settingsPath.extension().string()));
            utils::copyFile(settings.settingsPath, outSettingsPath);

            const fs::path outSettingsHashes = currentSavePathBase / (fmt::format("{0}_hashes{1}", currentTime, settings.settingsPath.extension().string()));
            settingHashes.push_back({ saveSettingsHash , saveSettings });
            utils::saveSettingHashes(outSettingsHashes, settingHashes);

            // data normalization (may be Scaler::NONE)
            utils::scale(img.data, dataNormalization);

            // compute image hierarchy
            ComputeHierarchy ch;
            ch.setSkipLevelSimilarities(skipLevelSimilarities);
            ch.init(img.data, img.height, img.width, ihs, lss, rws, nns, cs, cs_knn, cs_cc, cs_geo);
            ch.compute();

            // Save settings, stats and image hierarchy
            saveCurrentSettings(currentSavePath / "sph_settings.txt", nns, ihs, rws, lss);

            ch.getImageHierarchy()->writeStats((currentSavePath / "sph_stats_imh.txt").string());
            ch.getLevelSimilarities()->writeStats((currentSavePath / "sph_stats_ls.txt").string());

            const auto& h = ch.getImageHierarchy()->getHierarchy();
            const auto numLevels = h.getNumLevels();
            saveLevelImages(numLevels, h, img, currentSavePath); // , /*flipUd =*/ true

            // Save connected components
            Log::info("Saving connected components");
            std::shared_ptr<vi64> WCC;
            if (ch.getKnnDataLevel()->hasComponentsComputed())
            {
                if (ch.getKnnDataLevel()->hasComponentsConnected())
                {
                    Log::info("Saving connected components: Since WCC were connected, save black image");
                    WCC = std::make_shared<vi64>();
                    WCC->resize(img.data.getNumPoints());
                    std::fill(WCC->begin(), WCC->end(), 0);
                }
                else
                {
                    Log::info("Saving connected components: use previously computed connected components");
                    WCC = ch.getKnnDataLevelRef()->getConnectedComponentsRef();
                }
            }
            else
            {
                Log::info("Saving connected components: first recompute connected components... (since they were not connected earlier)");
                auto [numWCC, connectedComponents] = ch.getKnnDataLevelRef()->computeConnectedComponents();
                WCC = connectedComponents;
            }

            // Save component image (random component IDs)
            if (WCC->size() == static_cast<size_t>(img.width) * img.height)
                utils::saveSingleImage(*WCC, img.width, img.height, currentSavePath / "component.tiff"); // , /*flipUd =*/ true

            // Save component to pixel mapping
            utils::writeVecOfVecOfVecToBinary((currentSavePath / "MapFromLevelToBottom.bin").string(), h.mapFromLevelToPixel);
            utils::writeVecOfVecToBinary((currentSavePath / "MapFromBottomToLevel.bin").string(), h.mapFromPixelToLevel());

            utils::Logger::flush();

            if (!skipSaveStructuresToDisk)
            {
                Log::info("Saving additional data structures for evaluation...");
                const auto cachePathProbDists = currentSavePath / "ProbDists";
                const auto cachePathRandomWalks = currentSavePath / "RandomWalkSimilarities";
                for (int64_t level = 0; level < static_cast<int64_t>(h.getNumLevels()); level++)
                {
                    std::string fileName = cachePathProbDists.string() + std::to_string(level) + ".cache";
                    Log::info("Saving {}", fileName);
                    utils::writeSparseMatHDIToBinary(fileName, ch.getLevelSimilarities()->getProbDist(level));

                    fileName = cachePathRandomWalks.string() + std::to_string(level) + ".cache";
                    Log::info("Saving {}", fileName);
                    utils::writeSparseMatSPHToBinary(fileName, h.randomWalks[level]);
                }

            }

            // compute embedding
            for (size_t level = 0; level < numLevels; level++)
            {
                if (skipEmbeddingTSNE && skipEmbeddingUMAP)
                    break;

                Log::info("Compute embedding on level {}", level);

                ComputeEmbedding ce = {};

                const auto numComp = h.numComponentsOn(level);

                auto initEmbedding = [&ch, &settings, &h, &numComp, &initLevelEmbWithPrevious, &currentSavePath](const size_t level, ComputeEmbedding& ce, const std::string embType) {
                    if (level == 0)
                    {
                        const auto nnData = ch.getKnnDataLevelRef()->getNnData();
                        const auto& nNGraph = ch.getImageHierarchy()->getDataKnnGraph();

                        bool success = false;
                        std::vector<float> init = {};

                        if (settings.initEmbeddingDataLevel == utils::EmbeddingInit::PCA)
                        {
                            Log::info("Init {} with PCA", embType);
                            std::tie(init, success) = utils::pca(nnData.getData(), nnData.getNumDimensions());
                        }
                        else if (settings.initEmbeddingDataLevel == utils::EmbeddingInit::SPECTRAL)
                        {
                            Log::info("Init {} with spectral embedding", embType);
                            std::tie(init, success) = utils::spectralEmbedding(nNGraph);
                        }
                        else
                            Log::info("Init {} with random embedding", embType);

                        if (success)
                        {
                            utils::scaleEmbeddingToOne(init);
                            ce.initEmbedding(numComp, std::move(init));
                        }
                        else if (settings.initEmbeddingDataLevel != utils::EmbeddingInit::RANDOM)
                            Log::warn("Init could not be computed, fall back to random");
                    }
                    else if (initLevelEmbWithPrevious)
                    {
                        Log::info("Init based on previous embedding");
                        const std::string previousTsneSavePath = (currentSavePath / ("emb_" + embType + "_" + std::to_string(level - 1))).string() + ".bin";
                        std::vector<float> previousTsne;
                        utils::loadVecFromBinary(previousTsneSavePath, previousTsne);
                        std::vector<float> previousAveragePositions = utils::averageEmbeddingPositionOfChildren(h, previousTsne, level);
                        utils::scaleEmbeddingToOne(previousAveragePositions);
                        ce.initEmbedding(numComp, std::move(previousAveragePositions));
                    }

                    };

                if (normScheme == utils::NormalizationScheme::UMAP && !skipEmbeddingUMAP)
                {
                    Log::info("Compute UMAP");

                    if (numComp < 100)
                        ces.umap.numEpochs = 250;
                    else
                        ces.umap.numEpochs = 500;

                    if(level > 0 && initLevelEmbWithPrevious) // especially 500 iterations seems to destroy any alignment with previous emb
                      ces.umap.numEpochs = 175;

                    ce.setSettings(ces);
                    
                    initEmbedding(level, ce, "umap");

                    SparseMatHDI& probDist = ch.getLevelSimilaritiesRef()->getProbDistRef(level);

                    if (ch.getLevelSimilarities()->getProbDistIsSymmetric() == utils::NormalizationScheme::UMAP) {
                        Log::info("Symmetrize probDist for UMAP");
                        utils::symmetrizeUMAP(probDist);
                    }

                    utils::printSparseMatrixStats(utils::sparseMatrixStats(probDist), "Symmetric probDist");
                    
                    std::variant<const SparseMatHDI*, const utils::GraphBaseInterface*> umapInput = &probDist;

                    ce.computeUMAP(umapInput);

                    // save plots
                    std::string umapSavePath = (currentSavePath / ("emb_umap_" + std::to_string(level))).string();
                    Log::info("Saving umap embedding to {}", umapSavePath);
                    utils::writeVecToBinary(umapSavePath + ".bin", ce.getEmbedding());
                }

                if (normScheme == utils::NormalizationScheme::TSNE && !skipEmbeddingTSNE)
                {
                    Log::info("Compute t-SNE");

                    ces.tsne.gradientDescentType = static_cast<GradientDescentType>(0);  // GPUcompute is not always available;
                    ces.tsne.numIterations = 4000;

                    if (numComp < 100) {
                        ces.tsne.gradientDescentType = GradientDescentType::CPU;    // Ensure CPU gradient descent for very few points (GPU version is very slow there)
                        ces.tsne.numIterations = 500;
                    }
                    else if (numComp < 100'000) {
                        ces.tsne.numIterations = 1000;
                    }
                    else if (numComp < 200'000) {
                        ces.tsne.numIterations = 2000;
                    }

                    ces.tsne.perplexity = ch.getLevelSimilarities()->getPerplexities()[level];
                    ce.setSettings(ces);

                    initEmbedding(level, ce, "tsne");

                    SparseMatHDI& probDist = ch.getLevelSimilaritiesRef()->getProbDistRef(level);
                    
                    if (ch.getLevelSimilarities()->getProbDistIsSymmetric() == utils::NormalizationScheme::NONE) {
                        Log::info("Symmetrize probDist for t-SNE");
                        utils::symmetrizeTSNE(probDist);
                    }

                    utils::printSparseMatrixStats(utils::sparseMatrixStats(probDist), "Symmetric probDist");

                    ce.computeTSNE(&probDist);

                    // save plots
                    std::string tsneSavePath = (currentSavePath / ("emb_tsne_" + std::to_string(level))).string();
                    Log::info("Saving tsne embedding to {}", tsneSavePath);
                    utils::writeVecToBinary(tsneSavePath + ".bin", ce.getEmbedding());
                }

            }

            Log::info("Finished {0} of {1}", runID, numRuns);

        } // settingsCombinationsRandomWalks

    } // settingsCombinationsGeneral

    Log::info("Finished eval: {}", utils::getCurrentDateTimeHuman());
}

} // namespace sph::eval
