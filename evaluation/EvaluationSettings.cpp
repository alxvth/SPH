#include "EvaluationSettings.hpp"

#include <sph/utils/CommonDefinitions.hpp>

#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include <fmt/ostream.h>
#include <nlohmann/json.hpp>

namespace sph::eval
{
    namespace fs = std::filesystem;

    static void warnUnknownOption(const std::string& opt, const std::string& unknown)
    {
        fmt::println(std::cerr, "-- WARNING -- {0}: unknown option {1}. Returning default", opt, unknown);
    }

    static fs::path resolveRelativePath(const fs::path& input, const fs::path& baseFile)
    {
        // Only handle paths starting with "."
        if (!input.empty() && input.begin()->string() == ".") {
            return fs::weakly_canonical(baseFile.parent_path() / input);
        }

        return input;
    }

    template<>
    ImageFormat stringToSetting(const std::string& s)
    {
        if (s == "RGB")
            return eval::ImageFormat::RGB;
        if (s == "TIFFSTACK")
            return eval::ImageFormat::TIFFSTACK;

        warnUnknownOption("stringToImageFormat", s);

        return eval::ImageFormat::RGB;
    }

    template<>
    utils::RandomWalkHandling stringToSetting(const std::string& s)
    {
        if (s == "MERGE_RW_ONLY")
            return utils::RandomWalkHandling::MERGE_RW_ONLY;
        if (s == "MERGE_RW_NEW_WALKS")
            return utils::RandomWalkHandling::MERGE_RW_NEW_WALKS;
        if (s == "MERGE_RW_NEW_WALKS_AND_KNN")
            return utils::RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN;
        if (s == "MERGE_DATA_NEW_WALKS")
            return utils::RandomWalkHandling::MERGE_DATA_NEW_WALKS;

        warnUnknownOption("stringToRandomWalkHandling", s);

        return utils::RandomWalkHandling::MERGE_RW_ONLY;
    }

    template<>
    utils::RandomWalkReduction stringToSetting(const std::string& s)
    {
        if (s == "NONE")
            return utils::RandomWalkReduction::NONE;
        if (s == "PROPORTIONAL_COMPONENT_REDUCTION")
            return utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION;
        if (s == "PROPORTIONAL_HALF")
            return utils::RandomWalkReduction::PROPORTIONAL_HALF;
        if (s == "PROPORTIONAL_DOUBLE")
            return utils::RandomWalkReduction::PROPORTIONAL_DOUBLE;
        if (s == "CONSTANT")
            return utils::RandomWalkReduction::CONSTANT;
        if (s == "CONSTANT_LOW")
            return utils::RandomWalkReduction::CONSTANT_LOW;
        if (s == "CONSTANT_HIGH")
            return utils::RandomWalkReduction::CONSTANT_HIGH;

        warnUnknownOption("stringToRandomWalkReduction", s);

        return utils::RandomWalkReduction::NONE;
    }

    template<>
    utils::NeighConnection stringToSetting(const std::string& s)
    {
        if (s == "FOUR")
            return utils::NeighConnection::FOUR;
        if (s == "EIGHT")
            return utils::NeighConnection::EIGHT;

        warnUnknownOption("stringToNeighConnection", s);

        return utils::NeighConnection::FOUR;
    }

    template<>
    utils::EmbeddingInit stringToSetting(const std::string& s)
    {
        if (s == "RANDOM")
            return utils::EmbeddingInit::RANDOM;
        if (s == "PCA")
            return utils::EmbeddingInit::PCA;
        if (s == "SPECTRAL")
            return utils::EmbeddingInit::SPECTRAL;

        warnUnknownOption("stringToEmbeddingInit", s);

        return utils::EmbeddingInit::RANDOM;
    }

    template<>
    utils::ImportanceWeighting stringToSetting(const std::string& s)
    {
        if (s == "CONSTANT")
            return utils::ImportanceWeighting::CONSTANT;
        if (s == "LINEAR")
            return utils::ImportanceWeighting::LINEAR;
        if (s == "NORMAL")
            return utils::ImportanceWeighting::NORMAL;
        if (s == "ONLYLAST")
            return utils::ImportanceWeighting::ONLYLAST;
        if (s == "FIRST_VISIT")
            return utils::ImportanceWeighting::FIRST_VISIT;

        warnUnknownOption("stringToNeighConnection", s);

        return utils::ImportanceWeighting::CONSTANT;
    }

    template<>
    utils::NormalizationScheme stringToSetting(const std::string& s)
    {
        if (s == "TSNE")
            return utils::NormalizationScheme::TSNE;
        if (s == "LINEAR")
            return utils::NormalizationScheme::LINEAR;
        if (s == "UMAP")
            return utils::NormalizationScheme::UMAP;
        if (s == "NONE")
            return utils::NormalizationScheme::NONE;

        warnUnknownOption("stringToNeighConnection", s);

        return utils::NormalizationScheme::TSNE;
    }

    template<>
    utils::ComponentSim stringToSetting(const std::string& s)
    {
        if (s == "NEIGH_OVERLAP")
            return utils::ComponentSim::NEIGH_OVERLAP;
        if (s == "GEO_CENTROID")
            return utils::ComponentSim::GEO_CENTROID;
        if (s == "NEIGH_WALKS")
            return utils::ComponentSim::NEIGH_WALKS;
        if (s == "NEIGH_WALKS_SINGLE_OVERLAP")
            return utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP;
        if (s == "GEO_WALKS")
            return utils::ComponentSim::GEO_WALKS;
        if (s == "EUCLID_CENTROID")
            return utils::ComponentSim::EUCLID_CENTROID;

        warnUnknownOption("stringToComponentSim", s);

        return utils::ComponentSim::NEIGH_OVERLAP;
    }

    template<>
    utils::KnnMetric stringToSetting(const std::string& s)
    {
        if (s == "L2")
            return utils::KnnMetric::L2;
        if (s == "COSINE")
            return utils::KnnMetric::COSINE;
        if (s == "INNER_PRODUCT")
            return utils::KnnMetric::INNER_PRODUCT;

        warnUnknownOption("stringsToKnnMetric", s);

        return utils::KnnMetric::L2;
    }

    template<>
    utils::Scaler stringToSetting(const std::string& s)
    {
        if (s == "NONE")
            return utils::Scaler::NONE;
        if (s == "STANDARD")
            return utils::Scaler::STANDARD;
        if (s == "UNIFORM")
            return utils::Scaler::UNIFORM;
        if (s == "ROBUST")
            return utils::Scaler::ROBUST;

        warnUnknownOption("stringsToScaler", s);

        return utils::Scaler::NONE;
    }

    std::pair<EvalSettings, bool> readSettingsFromFile(const fs::path& settingsFile)
    {
        EvalSettings settings;

        if (!fs::exists(settingsFile))
        {
            fmt::println(std::cerr, "readSettingsFromFile: file {0} does not exists, using default settings", settingsFile.string());
            return { settings , false };
        }

        std::ifstream loadFile(settingsFile, std::ios::in);
        if (!loadFile.is_open())
        {
            fmt::println(std::cerr, "readSettingsFromFile: file {0} cannot be opened, using default settings", settingsFile.string());
            return { settings , false };
        }

        nlohmann::json settingsContent;
        loadFile >> settingsContent;

        // input
        settings.inputPath                  = resolveRelativePath(std::string(settingsContent["inputPath"]), settingsFile);
        settings.imgFormat                  = stringToSetting<ImageFormat>(settingsContent["imgFormat"]);
        settings.imageNames                 = settingsContent.at("imageNames").get<std::vector<std::string>>();

        settings.componentSim               = stringsToSettings<utils::ComponentSim>(settingsContent["componentSim"]);
        settings.dataInputNorm              = stringsToSettings<utils::Scaler>(settingsContent["dataInputNorm"]);

        // output
        settings.cachePathBase              = resolveRelativePath(std::string(settingsContent["cachePathBase"]), settingsFile);

        if(settingsContent.contains("saveSubFolder"))
          settings.saveSubFolder            = std::string(settingsContent["saveSubFolder"]);

        // spatial and data neighbors
        settings.neighborConnections        = stringsToSettings<utils::NeighConnection>(settingsContent["neighborConnection"]);
        settings.neighborSymmetries         = settingsContent.at("neighborSymmetries").get<std::vector<bool>>();
        settings.neighborConnectComponents  = settingsContent.at("neighborConnectComponents").get<std::vector<bool>>();
        settings.nKnns                      = settingsContent.at("nKnns").get<vui64>();
        settings.knnMetrics                 = stringsToSettings<utils::KnnMetric>(settingsContent["knnMetric"]);

        settings.dataDistNorm               = stringsToSettings<utils::NormalizationScheme>(settingsContent["dataDistNorm"]);

        // random walks
        settings.randomWalkNums             = settingsContent.at("randomWalkNums").get<vui64>();
        settings.randomWalkLens             = settingsContent.at("randomWalkLens").get<vui64>();
        settings.randomWalkStepWeight       = stringsToSettings<utils::ImportanceWeighting>(settingsContent["randomWalkStepWeight"]);
        settings.randomWalkHandling         = stringsToSettings<utils::RandomWalkHandling>(settingsContent["randomWalkHandling"]);
        settings.randomWalkReduction        = stringsToSettings<utils::RandomWalkReduction>(settingsContent["randomWalkReduction"]);
        settings.randomWalkWeightSize       = settingsContent.at("randomWalkWeightSize").get<std::vector<bool>>();
        settings.randomWalkPairSims         = settingsContent.at("randomWalkPairSims").get<std::vector<bool>>();

        // embedding
        settings.weightTransitionBySize     = settingsContent.at("weightTransitionBySize").get<std::vector<bool>>();
        settings.initEmbeddingDataLevel     = stringToSetting<utils::EmbeddingInit>(settingsContent["initEmbeddingDataLevel"]);

        // internal
        settings.settingsPath               = settingsFile;

        // Configure eval steps
        if (settingsContent.contains("skipLevelSimilarities"))
          settings.skipLevelSimilarities    = settingsContent["skipLevelSimilarities"];
        if (settingsContent.contains("skipEmbeddingTSNE"))
          settings.skipEmbeddingTSNE        = settingsContent["skipEmbeddingTSNE"];
        if (settingsContent.contains("skipEmbeddingUMAP"))
          settings.skipEmbeddingUMAP        = settingsContent["skipEmbeddingUMAP"];
        if (settingsContent.contains("skipExistingSettings"))
          settings.skipExistingSettings     = settingsContent["skipExistingSettings"];
        if (settingsContent.contains("skipSaveStructuresToDisk"))
          settings.skipSaveStructuresToDisk = settingsContent["skipSaveStructuresToDisk"];
        if (settingsContent.contains("initLevelEmbWithPrevious"))
          settings.initLevelEmbWithPrevious = settingsContent["initLevelEmbWithPrevious"];

        return { settings , true };
    }

    static std::string appendGeneralSettings(const std::string& saveSettingsIn, utils::ComponentSim compSim, const utils::NormalizationScheme& dataDistNorm, utils::Scaler normData, bool weightTransitionBySize, const EvalSettingNeighbors& setNeigh)
    {
        std::string saveSettingsOut = saveSettingsIn;

        saveSettingsOut += "compS" + std::to_string(utils::to_underlying(compSim));
        saveSettingsOut += "ddNor" + std::to_string(utils::to_underlying(dataDistNorm));
        saveSettingsOut += "nbCon" + std::to_string(utils::to_underlying(setNeigh.neighborConnection));
        saveSettingsOut += "nbSim" + std::to_string(static_cast<uint32_t>(setNeigh.neighborSymmetric));
        saveSettingsOut += "nbWCC" + std::to_string(static_cast<uint32_t>(setNeigh.neighborConnectComponent));
        saveSettingsOut += "knMet" + std::to_string(utils::to_underlying(setNeigh.knnMetric));
        saveSettingsOut += "nbKnn" + std::to_string(setNeigh.nKnns);
        saveSettingsOut += "dNorm" + std::to_string(utils::to_underlying(normData));
        saveSettingsOut += "wTrbS" + std::to_string(weightTransitionBySize);

        return saveSettingsOut;
    }

    static std::string appendRandomWalkSettings(const std::string& saveSettingsIn, const EvalSettingRandomWalks& setWalk)
    {
        std::string saveSettingsOut = saveSettingsIn;

        saveSettingsOut += "rwNum" + std::to_string(setWalk.randomWalkNum);
        saveSettingsOut += "rwLen" + std::to_string(setWalk.randomWalkLen);
        saveSettingsOut += "rwWei" + std::to_string(utils::to_underlying(setWalk.randomWalkWei));
        saveSettingsOut += "rwSei" + std::to_string(static_cast<uint32_t>(setWalk.randomWalkWeightSize));
        saveSettingsOut += "rwHan" + std::to_string(utils::to_underlying(setWalk.randomWalkHan));
        saveSettingsOut += "rwRed" + std::to_string(utils::to_underlying(setWalk.randomWalkRed));
        saveSettingsOut += "rwPai" + std::to_string(static_cast<uint32_t>(setWalk.randomWalkPairSims));

        return saveSettingsOut;
    }

    std::string getSettingsString(utils::ComponentSim compSim, utils::NormalizationScheme dataDistNorm, utils::Scaler normData, bool weightTransitionBySize, const EvalSettingNeighbors& setNeigh, const EvalSettingRandomWalks& setWalk)
    {
        const bool isRandomWalk = !onlyGeodesicSettings(compSim);

        std::string saveSettings = isRandomWalk ? "evalRW_" : "evalGEO_";

        saveSettings = appendGeneralSettings(saveSettings, compSim, dataDistNorm, normData, weightTransitionBySize, setNeigh);

        if(isRandomWalk)
            saveSettings = appendRandomWalkSettings(saveSettings, setWalk);

        return saveSettings;
    }

} // namespace sph::eval
