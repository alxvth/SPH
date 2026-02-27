#include "Settings.hpp"

#include "FileIO.hpp"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <type_traits>

#include <fmt/ranges.h>
#include <nlohmann/json.hpp>

namespace sph::utils {

    std::ostream& operator << (std::ostream& os, const sph::utils::KnnIndex& obj)
    {
        switch (obj)
        {
            case sph::utils::KnnIndex::BruteForce: 
                return os << "BruteForce";
            case sph::utils::KnnIndex::Flat: 
                return os << "Flat";
            case sph::utils::KnnIndex::IVFFlat: 
                return os << "IVFFlat";
            case sph::utils::KnnIndex::HNSW: 
                return os << "HNSW";
            case sph::utils::KnnIndex::HNSWSQ: 
                return os << "HNSWSQ";
            case sph::utils::KnnIndex::HNSW_IVFPQ: 
                return os <<"HNSW_IVFPQ";
        }

        return os << static_cast<std::underlying_type<sph::utils::KnnIndex>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::KnnMetric& obj)
    {
        switch (obj)
        {
            case sph::utils::KnnMetric::L2: 
                return os << "L2";
            case sph::utils::KnnMetric::COSINE: 
                return os << "COSINE";
            case sph::utils::KnnMetric::INNER_PRODUCT: 
                return os << "INNER_PRODUCT";
        }

        return os << static_cast<std::underlying_type<sph::utils::KnnMetric>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::ComponentSim& obj)
    {
        switch (obj)
        {
            case sph::utils::ComponentSim::NEIGH_OVERLAP: 
                 return os << "NEIGH_OVERLAP";
            case sph::utils::ComponentSim::GEO_CENTROID: 
                 return os << "GEO_CENTROID";
            case sph::utils::ComponentSim::NEIGH_WALKS: 
                 return os << "NEIGH_WALKS";
            case sph::utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP: 
                 return os << "NEIGH_WALKS_SINGLE_OVERLAP";
            case sph::utils::ComponentSim::GEO_WALKS: 
                 return os << "GEO_WALKS";
            case sph::utils::ComponentSim::EUCLID_CENTROID: 
                 return os << "EUCLID_CENTROID";
        }

        return os << static_cast<std::underlying_type<sph::utils::ComponentSim>::type>(obj);
    }

    std::ostream& operator<< (std::ostream& os, const sph::utils::ImportanceWeighting& obj)
    {
        std::string s;

        switch (obj)
        {
            case sph::utils::ImportanceWeighting::CONSTANT: 
                return os << "CONSTANT";
            case sph::utils::ImportanceWeighting::LINEAR: 
                return os << "LINEAR"; 
            case sph::utils::ImportanceWeighting::NORMAL: 
                return os << "NORMAL";
            case sph::utils::ImportanceWeighting::ONLYLAST: 
                return os << "ONLYLAST";
            case sph::utils::ImportanceWeighting::FIRST_VISIT: 
                return os << "FIRST_VISIT";
        }

        return os << static_cast<std::underlying_type<sph::utils::ImportanceWeighting>::type>(obj);
    }

    std::ostream& operator<< (std::ostream& os, const sph::utils::NormalizationScheme& obj)
    {
        switch (obj)
        {
            case sph::utils::NormalizationScheme::NONE: 
                return os << "NONE";
            case sph::utils::NormalizationScheme::TSNE: 
                return os << "TSNE";
            case sph::utils::NormalizationScheme::UMAP: 
                return os << "UMAP";
            case sph::utils::NormalizationScheme::LINEAR: 
                return os << "LINEAR";
        }

        return os << static_cast<std::underlying_type<sph::utils::NormalizationScheme>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::NeighConnection& obj)
    {
        switch (obj)
        {
            case sph::utils::NeighConnection::FOUR: 
                return os << "FOUR"; break;
            case sph::utils::NeighConnection::EIGHT: 
                return os << "EIGHT"; break;
        }

        return os << static_cast<std::underlying_type<sph::utils::NeighConnection>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::RandomWalkHandling& obj)
    {
        switch (obj)
        {
        case sph::utils::RandomWalkHandling::MERGE_RW_ONLY:
            return os << "MERGE_RW_ONLY"; break;
        case sph::utils::RandomWalkHandling::MERGE_RW_NEW_WALKS: 
            return os << "MERGE_RW_NEW_WALKS"; break;
        case sph::utils::RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN:
            return os << "MERGE_RW_NEW_WALKS_AND_KNN"; break;
        case sph::utils::RandomWalkHandling::MERGE_DATA_NEW_WALKS:
            return os << "MERGE_DATA_NEW_WALKS"; break;
        }

        return os << static_cast<std::underlying_type<sph::utils::RandomWalkHandling>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::RandomWalkReduction& obj)
    {
        switch (obj)
        {
        case sph::utils::RandomWalkReduction::NONE: 
            return os << "NONE"; break;
        case sph::utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION: 
            return os << "PROPORTIONAL_COMPONENT_REDUCTION"; break;
        case sph::utils::RandomWalkReduction::PROPORTIONAL_HALF:
            return os << "PROPORTIONAL_HALF"; break;
        case sph::utils::RandomWalkReduction::PROPORTIONAL_DOUBLE:
            return os << "PROPORTIONAL_DOUBLE"; break;
        case sph::utils::RandomWalkReduction::CONSTANT: 
            return os << "CONSTANT"; break;
        case sph::utils::RandomWalkReduction::CONSTANT_LOW: 
            return os << "CONSTANT_LOW"; break;
        case sph::utils::RandomWalkReduction::CONSTANT_HIGH: 
            return os << "CONSTANT_HIGH"; break;
        }

        return os << static_cast<std::underlying_type<sph::utils::RandomWalkReduction>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::Scaler& obj)
    {
        switch (obj)
        {
        case sph::utils::Scaler::NONE: 
            return os << "NONE"; break;
        case sph::utils::Scaler::STANDARD: 
            return os << "STANDARD"; break;
        case sph::utils::Scaler::UNIFORM: 
            return os << "UNIFORM"; break;
        case sph::utils::Scaler::ROBUST: 
            return os << "ROBUST"; break;
        }

        return os << static_cast<std::underlying_type<sph::utils::Scaler>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::EmbeddingInit& obj)
    {
        switch (obj)
        {
        case sph::utils::EmbeddingInit::RANDOM:
            return os << "RANDOM"; break;
        case sph::utils::EmbeddingInit::PCA: 
            return os << "PCA"; break;
        case sph::utils::EmbeddingInit::SPECTRAL: 
            return os << "SPECTRAL"; break;
        }

        return os << static_cast<std::underlying_type<sph::utils::EmbeddingInit>::type>(obj);
    }

    std::ostream& operator << (std::ostream& os, const sph::utils::NormType& obj)
    {
        switch (obj)
        {
        case sph::utils::NormType::ONEDIM:
            return os << "ONEDIM"; break;
        case sph::utils::NormType::TWODIM:
            return os << "TWODIM"; break;
        }

        return os << static_cast<std::underlying_type<sph::utils::NormType>::type>(obj);
    }

    void printSettings(const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss, const NearestNeighborsSettings& nns, const utils::RandomWalkSettings& rws)
    {
        Log::info("## ######## ##");
        Log::info("## Settings ##");
        Log::info("## ######## ##");

        printNearestNeighborsSettings(nns);
        printImageHierarchySettings(ihs);
        printLevelSimilaritiesSettings(lss);
        printRandomWalkSettings(rws);

        Log::info("## ######## ##");
    }

    void printNearestNeighborsSettings(const NearestNeighborsSettings& nns)
    {
        Log::info("## NearestNeighborsSettings ##");
        Log::info("numNearestNeighbors: {}", nns.numNearestNeighbors);
        Log::info("knnIndex: {}", nns.knnIndex);
        Log::info("knnMetric: {}", nns.knnMetric);
        Log::info("symmetricNeighbors: {}", nns.symmetricNeighbors);
        Log::info("computeConnectComponents: {}", nns.computeConnectComponents);
        Log::info("neighborConnectComponents: {}", nns.neighborConnectComponents);
        Log::info("L2squared: {}", nns.L2squared);
    }

    void printLevelSimilaritiesSettings(const LevelSimilaritiesSettings& lss)
    {
        Log::info("## LevelSimilaritiesSettings ##");
        Log::info("componentSim: {}", lss.componentSim);

        if (lss.ks.empty())
            Log::info("ks: empty (automatically populated if not explicitly set)");
        else
            Log::info("ks: {}", fmt::join(lss.ks, ", "));

        Log::info("componentLabels: {}", lss.componentLabels && !lss.componentLabels->empty());
        Log::info("forceComputeDistances: {}", lss.forceComputeDistances);
        Log::info("exactKnn: {}", lss.exactKnn);
        Log::info("randomWalkPairSims: {}", lss.randomWalkPairSims);
        Log::info("computeSymmetricProbDist: {}", lss.computeSymmetricProbDist);
        Log::info("normalizeProbDist: {}", lss.normalizeProbDist);
        Log::info("weightTransitionBySize: {}", lss.weightTransitionBySize);

        if (lss.levelToCompute == -1)
            Log::info("levelToCompute: all (== -1)");
        else
            Log::info("levelToCompute: {}", lss.levelToCompute);

    }

    void printImageHierarchySettings(const ImageHierarchySettings& ihs)
    {
        Log::info("## ImageHierarchySettings ##");
        Log::info("componentSim: {}", ihs.componentSim);
        Log::info("neighborConnection: {}", ihs.neighborConnection);
        Log::info("mergeMultiple: {}", ihs.mergeMultiple);
        Log::info("usePercentile: {}", ihs.usePercentile);
        Log::info("maxDist: {}", ihs.maxDist);
        Log::info("minNumComp: {}", ihs.minNumComp);
        Log::info("normKnnDistances: {}", ihs.normKnnDistances);
        Log::info("componentLabels: {}", ihs.componentLabels && !ihs.componentLabels->empty());
        Log::info("minReduction: {}", ihs.minReduction);
        Log::info("numGeodesicSamples: {}", ihs.numGeodesicSamples);
        Log::info("maxLevels: {}", ihs.maxLevels);
        Log::info("rwHandling: {}", ihs.rwHandling);
        Log::info("rwReduction: {}", ihs.rwReduction);
        Log::info("rwWeightMergeBySize: {}", ihs.rwWeightMergeBySize);
        Log::info("verbose: {}", ihs.verbose);
        Log::info("rwRandomWalkLengths: {}", ihs.rwRandomWalkLengths);
        Log::info("rwNormSim: {}", ihs.rwNormSim);
        Log::info("rwRemoveSelfSimAfterMerging: {}", ihs.rwRemoveSelfSimAfterMerging);
    }

    void printRandomWalkSettings(const utils::RandomWalkSettings& rws)
    {
        Log::info("## RandomWalkSettings ##");
        Log::info("numRandomWalks: {}", rws.numRandomWalks);
        Log::info("singleWalkLength: {}", rws.singleWalkLength);
        Log::info("minimumSingleWalkLength: {}", rws.minimumSingleWalkLength);
        Log::info("pruneValue: {}", rws.pruneValue);
        Log::info("importanceWeighting: {}", rws.importanceWeighting);
        Log::info("normalize: {}", rws.normalize);
        Log::info("removeDiagonal: {}", rws.removeDiagonal);
        Log::info("randomSeed: {}", rws.randomSeed);
    }

    std::pair<nlohmann::json, bool> loadJsonFromDisk(const std::string& fileName)
    {
        std::ifstream loadFile(fileName.c_str(), std::ios::in);

        if (!loadFile.is_open())
            return { nlohmann::json{}, false };

        Log::info("Loading " + fileName);

        // read a JSON file
        nlohmann::json parameters;
        loadFile >> parameters;

        return { parameters, true };
    }

    bool writeJsonToDisk(const std::string& fileName, const nlohmann::json& json)
    {
        std::ofstream saveFile(fileName, std::ios::out | std::ios::trunc);
        if (!utils::fileOpens(saveFile))
            return false;
        saveFile << std::setw(4) << json << std::endl;
        saveFile.close();

        return true;
    }

    void addToJson(const RandomWalkSettings& rws, nlohmann::json& json)
    {
        json["rws.numRandomWalks"]      = rws.numRandomWalks;
        json["rws.singleWalkLength"]    = rws.singleWalkLength;
        json["rws.minimumSingleWalkLength"]    = rws.minimumSingleWalkLength;
        json["rws.importanceWeighting"] = rws.importanceWeighting;
        json["rws.pruneValue"]          = rws.pruneValue;
        json["rws.pruneSteps"]          = rws.pruneSteps;
        json["rws.normalize"]           = rws.normalize;
        json["rws.removeDiagonal"]      = rws.removeDiagonal;
        json["rws.randomSeed"]          = rws.randomSeed;
    }

    void readFromJson(const nlohmann::json& json, RandomWalkSettings& rws)
    {
        rws.numRandomWalks      = json["rws.numRandomWalks"];
        rws.singleWalkLength    = json["rws.singleWalkLength"];
        rws.minimumSingleWalkLength = json["rws.minimumSingleWalkLength"];
        rws.importanceWeighting = json["rws.importanceWeighting"];
        rws.pruneValue          = json["rws.pruneValue"];
        rws.pruneSteps          = json["rws.pruneSteps"];
        rws.normalize           = json["rws.normalize"];
        rws.removeDiagonal      = json["rws.removeDiagonal"];
        rws.randomSeed          = json["rws.randomSeed"];
    }

    bool checkSettings(const nlohmann::json& json, const RandomWalkSettings& rws)
    {
        if (!checkEntry("rws.numRandomWalks", json, rws.numRandomWalks)) return false;
        if (!checkEntry("rws.singleWalkLength", json, rws.singleWalkLength)) return false;
        if (!checkEntry("rws.minimumSingleWalkLength", json, rws.minimumSingleWalkLength)) return false;
        if (!checkEntry("rws.pruneValue", json, rws.pruneValue)) return false;
        if (!checkEntry("rws.pruneSteps", json, rws.pruneSteps)) return false;
        if (!checkEntry("rws.importanceWeighting", json, static_cast<std::underlying_type_t<utils::ImportanceWeighting>>(rws.importanceWeighting))) return false;
        if (!checkEntry("rws.normalize", json, rws.normalize)) return false;
        if (!checkEntry("rws.removeDiagonal", json, rws.removeDiagonal)) return false;
        if (!checkEntry("rws.randomSeed", json, rws.randomSeed)) return false;
        // parallel is not checked

        return true;
    }

    void addToJson(const NearestNeighborsSettings& nns, nlohmann::json& json)
    {
        json["nns.numNearestNeighbors"]         = nns.numNearestNeighbors;
        json["nns.knnMetric"]                   = nns.knnMetric;
        json["nns.knnIndex"]                    = nns.knnIndex;
        json["nns.L2squared"]                   = nns.L2squared;
        json["nns.computeConnectComponents"]    = nns.computeConnectComponents;
        json["nns.neighborConnectComponents"]   = nns.neighborConnectComponents;
        json["nns.symmetricNeighbors"]          = nns.symmetricNeighbors;
    }

    void addToJson(const ImageHierarchySettings& ihs, nlohmann::json& json)
    {
        json["ihs.minNumComp"]          = ihs.minNumComp;
        json["ihs.mergeMultiple"]       = ihs.mergeMultiple;
        json["ihs.usePercentile"]       = ihs.usePercentile;
        json["ihs.minReduction"]        = ihs.minReduction;
        json["ihs.maxDist"]             = static_cast<double>(ihs.maxDist);
        json["ihs.componentSim"]        = ihs.componentSim;
        json["ihs.normKnnDistances"]    = ihs.normKnnDistances;
        json["ihs.neighborConnection"]  = ihs.neighborConnection;
        json["ihs.rwHandling"]          = ihs.rwHandling;
        json["ihs.rwReduction"]         = ihs.rwReduction;
        json["ihs.maxLevels"]           = ihs.maxLevels;
        json["ihs.componentLabels"]     = ihs.componentLabels && !ihs.componentLabels->empty(); // is stored in NearestNeighbors
        json["ihs.numGeodesicSamples"]  = ihs.numGeodesicSamples;
        json["ihs.rwWeightMergeBySize"] = ihs.rwWeightMergeBySize;
        json["ihs.rwRandomWalkLengths"] = ihs.rwRandomWalkLengths;
        json["ihs.rwNormSim"]           = ihs.rwNormSim;
        json["ihs.rwRemoveSelfSimAfterMerging"] = ihs.rwRemoveSelfSimAfterMerging;
    }

    void addToJson(const LevelSimilaritiesSettings& lss, nlohmann::json& json)
    {
        json["lss.ks"]                          = lss.ks;
        json["lss.componentSim"]                = lss.componentSim;
        json["lss.forceComputeDistances"]       = lss.forceComputeDistances;
        json["lss.exactKnn"]                    = lss.exactKnn;
        json["lss.levelToCompute"]              = lss.levelToCompute;
        json["lss.randomWalkPairSims"]          = lss.randomWalkPairSims;
        json["lss.weightTransitionBySize"]      = lss.weightTransitionBySize;
        json["lss.computeSymmetricProbDist"]    = lss.computeSymmetricProbDist;
        json["lss.normalizeProbDist"]           = lss.normalizeProbDist;
        json["lss.componentLabels"]             = lss.componentLabels && !lss.componentLabels->empty(); // is stored in NearestNeighbors
    }

    void readFromJson(const nlohmann::json& json, NearestNeighborsSettings& nns)
    {
        nns.numNearestNeighbors         = json["nns.numNearestNeighbors"];
        nns.knnMetric                   = json["nns.knnMetric"];
        nns.knnIndex                    = json["nns.knnIndex"];
        nns.L2squared                   = json["nns.L2squared"];
        nns.computeConnectComponents    = json["nns.computeConnectComponents"];
        nns.neighborConnectComponents   = json["nns.neighborConnectComponents"];
        nns.symmetricNeighbors          = json["nns.symmetricNeighbors"];
    }

    void readFromJson(const nlohmann::json& json, ImageHierarchySettings& ihs)
    {
        ihs.minNumComp          = json["ihs.minNumComp"];
        ihs.mergeMultiple       = json["ihs.mergeMultiple"];
        ihs.usePercentile       = json["ihs.usePercentile"];
        ihs.minReduction        = json["ihs.minReduction"];
        ihs.maxDist             = json["ihs.maxDist"];
        ihs.componentSim        = json["ihs.componentSim"];
        ihs.normKnnDistances    = json["ihs.normKnnDistances"];
        ihs.neighborConnection  = json["ihs.neighborConnection"];
        ihs.rwHandling          = json["ihs.rwHandling"];
        ihs.rwReduction         = json["ihs.rwReduction"];
        ihs.maxLevels           = json["ihs.maxLevels"];
        ihs.numGeodesicSamples  = json["ihs.numGeodesicSamples"];
        ihs.rwWeightMergeBySize = json["ihs.rwWeightMergeBySize"];
        ihs.rwRandomWalkLengths = json.at("ihs.rwRandomWalkLengths").get<vui64>();
        ihs.rwNormSim           = json["ihs.rwNormSim"];
        ihs.rwRemoveSelfSimAfterMerging = json["ihs.rwRemoveSelfSimAfterMerging"];
        //ihs.componentLabels
    }

    void readFromJson(const nlohmann::json& json, LevelSimilaritiesSettings& lss)
    {
        lss.ks                          = json.at("lss.ks").get<vi64>();
        lss.componentSim                = json["lss.componentSim"];
        lss.forceComputeDistances       = json["lss.forceComputeDistances"];
        lss.exactKnn                    = json["lss.exactKnn"];
        lss.levelToCompute              = json["lss.levelToCompute"];
        lss.randomWalkPairSims          = json["lss.randomWalkPairSims"];
        lss.weightTransitionBySize      = json["lss.weightTransitionBySize"];
        lss.computeSymmetricProbDist    = json["lss.computeSymmetricProbDist"];
        lss.normalizeProbDist           = json["lss.normalizeProbDist"];
    }

    bool checkSettings(const nlohmann::json& json, const NearestNeighborsSettings& nns)
    {
        if (!checkEntry("nns.numNearestNeighbors", json, nns.numNearestNeighbors)) return false;
        if (!checkEntry("nns.knnMetric", json, static_cast<std::underlying_type_t<utils::KnnMetric>>(nns.knnMetric))) return false;
        if (!checkEntry("nns.knnIndex", json, static_cast<std::underlying_type_t<utils::KnnIndex>>(nns.knnIndex))) return false;
        if (!checkEntry("nns.L2squared", json, nns.L2squared)) return false;
        if (!checkEntry("nns.computeConnectComponents", json, nns.computeConnectComponents)) return false;
        if (!checkEntry("nns.neighborConnectComponents", json, nns.neighborConnectComponents)) return false;
        if (!checkEntry("nns.symmetricNeighbors", json, nns.symmetricNeighbors)) return false;

        return true;
    }

    bool checkSettings(const nlohmann::json& json, const ImageHierarchySettings& ihs)
    {
        if (!checkEntry("ihs.componentSim", json, static_cast<std::underlying_type_t<utils::ComponentSim>>(ihs.componentSim))) return false;
        if (!checkEntry("ihs.neighborConnection", json, static_cast<std::underlying_type_t<utils::NeighConnection>>(ihs.neighborConnection))) return false;
        if (!checkEntry("ihs.mergeMultiple", json, ihs.mergeMultiple)) return false;
        if (!checkEntry("ihs.usePercentile", json, ihs.usePercentile)) return false;
        if (!checkEntry("ihs.maxDist", json, ihs.maxDist)) return false;
        if (!checkEntry("ihs.minNumComp", json, ihs.minNumComp)) return false;
        if (!checkEntry("ihs.componentLabels", json, ihs.componentLabels && !ihs.componentLabels->empty())) return false;
        if (!checkEntry("ihs.minReduction", json, ihs.minReduction)) return false;
        if (!checkEntry("ihs.numGeodesicSamples", json, ihs.numGeodesicSamples)) return false;
        if (!checkEntry("ihs.maxLevels", json, ihs.maxLevels)) return false;
        // verbose setting is not checked
        if (!checkEntry("ihs.normKnnDistances", json, static_cast<std::underlying_type_t<utils::NormalizationScheme>>(ihs.normKnnDistances))) return false;
        if (!checkEntry("ihs.rwHandling", json, static_cast<std::underlying_type_t<utils::RandomWalkHandling>>(ihs.rwHandling))) return false;
        if (!checkEntry("ihs.rwReduction", json, static_cast<std::underlying_type_t<utils::RandomWalkReduction>>(ihs.rwReduction))) return false;
        if (!checkEntry("ihs.rwWeightMergeBySize", json, ihs.rwWeightMergeBySize)) return false;
        // do not check rwRandomWalkLengths: this vec is empty upon start but not empty in the cache
        // and will be populated based on rws.singleWalkLength and ihs.rwReduction
        //if (!checkEntry("ihs.rwRandomWalkLengths", json, ihs.rwRandomWalkLengths)) return false;
        if (!checkEntry("ihs.rwNormSim", json, ihs.rwNormSim)) return false;
        if (!checkEntry("ihs.rwRemoveSelfSimAfterMerging", json, ihs.rwRemoveSelfSimAfterMerging)) return false;

        return true;
    }

    bool checkSettings(const nlohmann::json& json, const LevelSimilaritiesSettings& lss)
    {
        if (!checkEntry("lss.componentSim", json, static_cast<std::underlying_type_t<utils::ComponentSim>>(lss.componentSim))) return false;
        if (!checkEntry("lss.ks", json, lss.ks)) return false;
        if (!checkEntry("lss.exactKnn", json, lss.exactKnn)) return false;
        if (!checkEntry("lss.componentLabels", json, lss.componentLabels && !lss.componentLabels->empty())) return false;
        if (!checkEntry("lss.forceComputeDistances", json, lss.forceComputeDistances)) return false;
        if (!checkEntry("lss.levelToCompute", json, lss.levelToCompute)) return false;
        if (!checkEntry("lss.randomWalkPairSims", json, lss.randomWalkPairSims)) return false;
        if (!checkEntry("lss.weightTransitionBySize", json, lss.weightTransitionBySize)) return false;
        if (!checkEntry("lss.normalizeProbDist", json, static_cast<std::underlying_type_t<utils::ComponentSim>>(lss.normalizeProbDist))) return false;
        if (!checkEntry("lss.computeSymmetricProbDist", json, static_cast<std::underlying_type_t<utils::ComponentSim>>(lss.computeSymmetricProbDist))) return false;

        return true;
    }

} // sph::utils
