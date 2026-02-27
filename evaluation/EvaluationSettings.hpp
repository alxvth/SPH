#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <sph/utils/Settings.hpp>

namespace sph::eval
{
    enum class ImageFormat {
        RGB = 0,
        TIFFSTACK = 1,
    };

    struct EvalSettings {
        // input
        std::filesystem::path inputPath = "images";
        ImageFormat imgFormat = ImageFormat::RGB; // eval::ImageFormat::TIFFSTACK
        std::vector<std::string> imageNames = { }; // e.g. { "bus.jpg", "umbrellas.jpg", "bike.jpg" }
        std::vector<utils::Scaler> dataInputNorm = { utils::Scaler::NONE, utils::Scaler::STANDARD, utils::Scaler::UNIFORM, utils::Scaler::ROBUST };

        std::vector<utils::ComponentSim> componentSim = { utils::ComponentSim::NEIGH_WALKS, utils::ComponentSim::GEO_CENTROID };

        // output
        std::filesystem::path cachePathBase = "evaluation";
        std::filesystem::path saveSubFolder = "";

        // spatial and data neighbors
        std::vector<utils::NeighConnection> neighborConnections = { utils::NeighConnection::EIGHT, utils::NeighConnection::FOUR };
        std::vector<bool> neighborSymmetries = { false, true };
        std::vector<bool> neighborConnectComponents = { false, true };
        std::vector<uint64_t> nKnns = { 10, 50, 100, 500, 1000 };
        std::vector<utils::KnnMetric> knnMetrics = { utils::KnnMetric::L2 };

        // random walks
        std::vector<utils::RandomWalkHandling> randomWalkHandling = { utils::RandomWalkHandling::MERGE_RW_ONLY };
        std::vector<uint64_t> randomWalkNums = { 10, 25, 50, 100, 150, 200, 250, 500, 1000 };
        std::vector<uint64_t> randomWalkLens = { 10, 25, 50, 100, 150, 200, 250, 500, 1000 };
        std::vector<utils::ImportanceWeighting> randomWalkStepWeight = { utils::ImportanceWeighting::CONSTANT, utils::ImportanceWeighting::LINEAR, utils::ImportanceWeighting::NORMAL, utils::ImportanceWeighting::ONLYLAST };
        std::vector<utils::RandomWalkReduction> randomWalkReduction = { utils::RandomWalkReduction::CONSTANT, utils::RandomWalkReduction::NONE, utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION };
        std::vector<bool> randomWalkWeightSize = { true, false };
        std::vector<bool> randomWalkPairSims = { true, false };
        
        std::vector<utils::NormalizationScheme> dataDistNorm = { utils::NormalizationScheme::TSNE, utils::NormalizationScheme::UMAP};

        // embedding
        utils::EmbeddingInit initEmbeddingDataLevel = utils::EmbeddingInit::RANDOM;
        std::vector<bool> weightTransitionBySize = { true, false };

        // internal
        std::filesystem::path settingsPath = std::filesystem::path{};

        // Configure eval steps
        bool skipLevelSimilarities    = false;
        bool skipEmbeddingTSNE        = false;
        bool skipEmbeddingUMAP        = false;
        bool skipExistingSettings     = true;
        bool skipSaveStructuresToDisk = true;
        bool initLevelEmbWithPrevious = false;

    };

    struct EvalSettingIO {
        std::filesystem::path inputPath = "images";
        ImageFormat imgFormat = ImageFormat::RGB;
        std::string imageName = ""; // e.g. "bus.jpg"
        std::filesystem::path cachePathBase = "evaluation";
        utils::Scaler dataInputNorm = utils::Scaler::NONE;
    };

    struct EvalSettingNeighbors {
        utils::NeighConnection neighborConnection = utils::NeighConnection::EIGHT;
        bool neighborSymmetric = false;
        bool neighborConnectComponent = false;
        uint64_t nKnns = 0;
        utils::KnnMetric knnMetric = utils::KnnMetric::L2;
    };

    struct EvalSettingRandomWalks {
        uint64_t randomWalkNum = 0;
        uint64_t randomWalkLen = 0;
        utils::ImportanceWeighting randomWalkWei = utils::ImportanceWeighting::CONSTANT;
        bool randomWalkWeightSize = true;
        bool randomWalkPairSims = false;
        utils::RandomWalkHandling randomWalkHan = utils::RandomWalkHandling::MERGE_RW_ONLY;
        utils::RandomWalkReduction randomWalkRed = utils::RandomWalkReduction::NONE;
    };

    template<typename T>
    T stringToSetting(const std::string& settingString);

    template<> ImageFormat stringToSetting<>(const std::string& settingString);
    template<> utils::RandomWalkHandling stringToSetting<>(const std::string& settingString);
    template<> utils::RandomWalkReduction stringToSetting<>(const std::string& settingString);
    template<> utils::NeighConnection stringToSetting<>(const std::string& settingString);
    template<> utils::ImportanceWeighting stringToSetting<>(const std::string& settingString);
    template<> utils::NormalizationScheme stringToSetting<>(const std::string& settingString);
    template<> utils::ComponentSim stringToSetting<>(const std::string& settingString);
    template<> utils::KnnMetric stringToSetting<>(const std::string& settingString);
    template<> utils::Scaler stringToSetting<>(const std::string& settingString);
    template<> utils::EmbeddingInit stringToSetting<>(const std::string& settingString);

    template<typename T>
    auto stringsToSettings(const std::vector<std::string>& vs) -> std::vector<T>
    {
        std::vector<T> vec;
        vec.reserve(vs.size());

        for (const auto& settingString : vs)
            vec.push_back(stringToSetting<T>(settingString));

        return vec;
    }

    inline bool onlyGeodesicSettings(const utils::ComponentSim& cs)
    {
        return cs == utils::ComponentSim::GEO_CENTROID || cs == utils::ComponentSim::EUCLID_CENTROID;
    }

    inline size_t numRandomWalkSettingComponentSim(const std::vector<utils::ComponentSim>& css)
    {
        size_t res = 0;

        for (const utils::ComponentSim cs : css) {
            if (!onlyGeodesicSettings(cs)) {
                res++;
            }
        }

        return res;
    }

    std::pair<EvalSettings, bool> readSettingsFromFile(const std::filesystem::path& settingsFile);

    std::string getSettingsString(utils::ComponentSim compSim, const utils::NormalizationScheme dataDistNorm, const utils::Scaler normData, bool weightTransitionBySize, const EvalSettingNeighbors& setNeigh, const EvalSettingRandomWalks& setWalk);

}
