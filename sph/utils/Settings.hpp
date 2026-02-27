#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <fmt/ostream.h>
#include <nlohmann/json.hpp>

#include "CommonDefinitions.hpp"
#include "Logger.hpp"

// Adding a setting?
// - Update operator<<
// - Update print* functions
// - Update saving to json
// - Update loading from json
// - Update checkSettings

namespace sph::utils {

    enum class Scaler : uint32_t {
        NONE,           // do nothing
        STANDARD,       // centers data around mean with unit standard deviation using z = (x - u) / s [channel-wise]
        UNIFORM,        // normalizes values to [0, 1] [channel-wise]
        ROBUST,         // clamps data to 95% and normalizes values to [0, 1] [globally]
    };

    // see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    enum class KnnIndex : uint32_t {
        BruteForce,     // exact -> faiss::HeapArray,
        Flat,           // exact -> faiss::IndexFlat, like BruteForce but copies data
        IVFFlat,        // approximate -> faiss::IndexIVFFlat
        HNSW,           // approximate -> faiss::IndexHNSWFlat
        HNSWSQ,         // approximate -> faiss::IndexHNSWSQ with faiss::ScalarQuantizer::QuantizerType::QT_8bit
        HNSW_IVFPQ      // approximate -> faiss::IndexIVFPQ with faiss::IndexHNSWFlat
    };

    enum class KnnMetric : uint32_t {
        L2,
        COSINE,
        INNER_PRODUCT
    };

    enum class ComponentSim : uint32_t {
        NEIGH_OVERLAP,
        GEO_CENTROID,
        NEIGH_WALKS,                    // Standard, uses Bhattacharyya coefficient
        NEIGH_WALKS_SINGLE_OVERLAP,
        GEO_WALKS,
        EUCLID_CENTROID,
    };

    enum class ImportanceWeighting : uint32_t {
        CONSTANT,       // each step is weighted the same, with 1
        LINEAR,         // weight is 1 - (step / (totalNumSteps + 1)), e.g.: 1, .909, .818, .727, .636, .545, .455 ,.363, .273, .182
        NORMAL,         // weight decreases wrt to gaussian distribution., e.g.: 1, .956, .835, .667, .487, .325, .198 ,.110, .056, .026
        ONLYLAST,       // weights only the final point
        FIRST_VISIT,
    };

    enum class NormalizationScheme : uint32_t {
        NONE,           // do nothing
        TSNE,           // Gaussian kernel as in t-SNE
        UMAP,           // Exponential kernel as in UMAP
        LINEAR,         // Simple linear normalization
    };

    enum class RandomWalkHandling : uint32_t {
        MERGE_RW_ONLY,
        MERGE_RW_NEW_WALKS,
        MERGE_RW_NEW_WALKS_AND_KNN,
        MERGE_DATA_NEW_WALKS
    };

    enum class RandomWalkReduction : uint32_t {
        NONE,
        PROPORTIONAL_COMPONENT_REDUCTION,
        PROPORTIONAL_HALF,
        PROPORTIONAL_DOUBLE,
        CONSTANT,
        CONSTANT_LOW,
        CONSTANT_HIGH
    };

    struct RandomWalkSettings
    {
        uint64_t numRandomWalks                 = 90;
        uint64_t singleWalkLength               = 15;             // used as initial walk length, might be reduced based on ihs.rwReduction
        uint64_t minimumSingleWalkLength        = 5;              // no reduction below this value
        float pruneValue                        = 0;              // remove entries below this value
        uint64_t pruneSteps                     = 0;              // computes pruneValue based on this and importanceWeighting. See utils::doRandomWalks
        ImportanceWeighting importanceWeighting = ImportanceWeighting::CONSTANT;
        bool normalize                          = true;
        bool removeDiagonal                     = true;           // if diagonal is the only element, it'll be kept. Would set self-hit random walk result to 0.
        uint64_t randomSeed                     = 1;
        bool parallel                           = true;           // this is to ensure reproducible tests
    };

    enum class NeighConnection : uint32_t {
        FOUR,
        EIGHT,
    };

    enum class EmbeddingInit : uint32_t {
        RANDOM,
        PCA,
        SPECTRAL,
    };

    enum class NormType : uint32_t {
        ONEDIM,
        TWODIM,
    };


} // namespace sph::utils

namespace sph {

    struct CacheSettings {
        std::string path               = "";
        std::string fileName           = "";
        bool cacheActive               = false;
        bool ignoreSubfolder           = false;
        std::string customSubfolder    = "";
    };

    struct NearestNeighborsSettings
    {
        uint64_t numNearestNeighbors   = 0;
        utils::KnnIndex knnIndex       = utils::KnnIndex::Flat;
        utils::KnnMetric knnMetric     = utils::KnnMetric::L2;
        bool symmetricNeighbors        = false;
        bool computeConnectComponents  = false;
        bool neighborConnectComponents = false;
        bool L2squared                 = false;
    };

    struct ImageHierarchySettings
    {
        utils::ComponentSim componentSim                    = utils::ComponentSim::NEIGH_OVERLAP;
        utils::NeighConnection neighborConnection           = utils::NeighConnection::FOUR;
        bool mergeMultiple                                  = false;
        bool usePercentile                                  = true;
        float maxDist                                       = 0;   // -1.f indicates to always merge
        int64_t minNumComp                                  = 1;
        std::shared_ptr<vi64> componentLabels               = nullptr;
        float minReduction                                  = 99.99f;
        size_t numGeodesicSamples                           = std::numeric_limits<size_t>::max(); // used in geodesic distance computation, a reasonable value might be 100
        int64_t maxLevels                                   = -1;
        bool verbose                                        = false;
        utils::RandomWalkHandling rwHandling                = utils::RandomWalkHandling::MERGE_RW_ONLY;
        utils::RandomWalkReduction rwReduction              = utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION;
        utils::NormalizationScheme normKnnDistances         = utils::NormalizationScheme::TSNE;
        bool rwWeightMergeBySize                            = true;
        vui64 rwRandomWalkLengths                           = {};
        utils::NormType rwNormSim                           = utils::NormType::ONEDIM;
        bool rwRemoveSelfSimAfterMerging                    = true;    // so that the self-transition probability for the next random walks is 0. Does NOT apply to RandomWalkHandling::MERGE_RW_ONLY

        inline bool isAlwaysMerge() const { return maxDist  == -1.f; }
    };

    struct LevelSimilaritiesSettings
    {
        utils::ComponentSim componentSim                    = utils::ComponentSim::NEIGH_OVERLAP;
        vi64 ks                                             = {};   // automatically populated if not explicitly set
        bool exactKnn                                       = false;
        std::shared_ptr<vi64> componentLabels               = nullptr;
        bool forceComputeDistances                          = false;   // Not all similarity setting require this stage but you can force the population the _similaritiesVector
        int64_t levelToCompute                              = -1; // -1 indicates all levels
        bool randomWalkPairSims                             = true; // whether to compute pair-wise sims based on random walk distribution, or just the largest random walk transition values as sims
        bool weightTransitionBySize                         = false; // whether to weight transition by size
        utils::NormalizationScheme normalizeProbDist        = utils::NormalizationScheme::TSNE; // default is to symmetrize for t-SNE
        utils::NormalizationScheme computeSymmetricProbDist = utils::NormalizationScheme::TSNE; // default is to symmetrize for t-SNE
    };

} // namespace sph::utils

/// ////////////// ///
/// printer helper ///
/// ////////////// ///

namespace sph::utils {
    std::ostream& operator << (std::ostream& os, const KnnIndex& obj);
    std::ostream& operator << (std::ostream& os, const KnnMetric& obj);
    std::ostream& operator << (std::ostream& os, const ComponentSim& obj);
    std::ostream& operator << (std::ostream& os, const ImportanceWeighting& obj);
    std::ostream& operator << (std::ostream& os, const NormalizationScheme& obj);
    std::ostream& operator << (std::ostream& os, const NeighConnection& obj);
    std::ostream& operator << (std::ostream& os, const RandomWalkHandling& obj);
    std::ostream& operator << (std::ostream& os, const RandomWalkReduction& obj);
    std::ostream& operator << (std::ostream& os, const EmbeddingInit& obj);
    std::ostream& operator << (std::ostream& os, const NormType& obj);
    std::ostream& operator << (std::ostream& os, const Scaler& obj);
} // sph::utils

// https://github.com/fmtlib/fmt/blob/11.1.3/doc/api.md#stdostream-support
// for gcc the << operator has to be overloaded in the same namespace as the enum definition
template <> struct fmt::formatter<sph::utils::KnnIndex> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::KnnMetric> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::ComponentSim> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::ImportanceWeighting> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::NormalizationScheme> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::NeighConnection> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::RandomWalkHandling> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::RandomWalkReduction> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::EmbeddingInit> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::NormType> : fmt::ostream_formatter {};
template <> struct fmt::formatter<sph::utils::Scaler> : fmt::ostream_formatter {};

namespace sph::utils {
    void printSettings(const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss, const NearestNeighborsSettings& nns, const utils::RandomWalkSettings& rws);
    void printNearestNeighborsSettings(const NearestNeighborsSettings& nns);
    void printLevelSimilaritiesSettings(const LevelSimilaritiesSettings& lss);
    void printImageHierarchySettings(const ImageHierarchySettings& ihs);
    void printRandomWalkSettings(const utils::RandomWalkSettings& rws);
} // sph::utils

/// ///////// ///
/// IO helper ///
/// ///////// ///

namespace sph::utils {

    std::pair<nlohmann::json, bool> loadJsonFromDisk(const std::string& fileName);
    bool writeJsonToDisk(const std::string& fileName, const nlohmann::json& json);

    void addToJson(const NearestNeighborsSettings& nns, nlohmann::json& json);
    void addToJson(const ImageHierarchySettings& ihs, nlohmann::json& json);
    void addToJson(const LevelSimilaritiesSettings& lss, nlohmann::json& json);
    void addToJson(const RandomWalkSettings& rws, nlohmann::json& json);

    void readFromJson(const nlohmann::json& json, NearestNeighborsSettings& nns);
    void readFromJson(const nlohmann::json& json, ImageHierarchySettings& ihs);
    void readFromJson(const nlohmann::json& json, LevelSimilaritiesSettings& lss);
    void readFromJson(const nlohmann::json& json, RandomWalkSettings& rws);

    bool checkSettings(const nlohmann::json& json, const NearestNeighborsSettings& nns);
    bool checkSettings(const nlohmann::json& json, const ImageHierarchySettings& ihs);
    bool checkSettings(const nlohmann::json& json, const LevelSimilaritiesSettings& lss);
    bool checkSettings(const nlohmann::json& json, const RandomWalkSettings& rws);

    template<typename T>
    bool checkEntry(const std::string& paramName, const nlohmann::json& json, const T& localParam) {
        if (!json.contains(paramName))
        {
            Log::warn("Cache does not contain {0}", paramName);
            return false;
        }

        const auto& storedParam = json.at(paramName);

        if (storedParam != localParam) {
            Log::info("{0} ({1}) does not match cache ({2}). Cannot load cache.",
                paramName,
                fmt::to_string(localParam),
                storedParam.dump());
            return false;
        }

        return true;
    }

    template<typename Enum>
    auto to_underlying(Enum setting) -> std::underlying_type_t<Enum>
    {
        return static_cast<std::underlying_type_t<Enum>>(setting);
    }

} // sph::utils
