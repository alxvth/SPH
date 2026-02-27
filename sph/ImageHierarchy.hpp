#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "utils/Cacheable.hpp"
#include "utils/CommonDefinitions.hpp"
#include "utils/Data.hpp"
#include "utils/Graph.hpp"
#include "utils/GraphBoost.hpp"
#include "utils/Hierarchy.hpp"
#include "utils/ImageHelper.hpp"
#include "utils/Settings.hpp"

namespace sph {

    /// /////////////// ///
    /// Image Hierarchy ///
    /// /////////////// ///

    struct ImageHierarchyStats
    {
        vui64   zeroSimilarityCount = {}; // per level
        vui64   forcedMergeCount = {}; // per level
        vf32    reductionRates = {};
        vf64    rwSparsities = {};
        vf64    mergedDataSparsities = {};
        vui64   numComponents = {};
        vui64   notMergedComponents = {}; // number of components that were not merged
    };

    /**
     * Image Hierarchy class
     */
    class ImageHierarchy : public utils::Cacheable
    {
    public:
        ImageHierarchy();
        ImageHierarchy(const utils::Graph& dataKnnGraph, const utils::Data& data, int64_t rows, int64_t cols, bool graphHasWCC = false);
        ImageHierarchy(const utils::GraphView& dataKnnGraph, const utils::DataView& data, int64_t rows, int64_t cols, bool graphHasWCC = false);
        ~ImageHierarchy() = default;

        void compute(const std::optional<ImageHierarchySettings>& ihs = std::nullopt, const std::optional<utils::RandomWalkSettings>& rws = std::nullopt);

    public: // Setter
        void setNeighConnection(utils::NeighConnection con) {
            _ihs.neighborConnection = con;
            _neighConnection.neighborConnection = con;
        }
        void setComponentSim(utils::ComponentSim cs) {
            _ihs.componentSim = cs;
        }
        void setMergeMultipleComponents(bool mergeMultiple) {
            _ihs.mergeMultiple = mergeMultiple;
        }
        void setUsePercentile(bool usePercentile) {
            _ihs.usePercentile = usePercentile;
        }
        void setMinimumNumComponents(int64_t minNumComp) {
            _ihs.minNumComp = minNumComp;
        }
        void setMaximumDistance(float maxDist) {
            _ihs.maxDist = maxDist;
        };
        void setNumGeodesicSamples(size_t numGeodesicSamples) {
            _ihs.numGeodesicSamples = numGeodesicSamples;
        };
        void setConnectedComponents(std::shared_ptr<vi64> componentLabels) {
            _ihs.componentLabels = componentLabels;
        };
        void setNormInputDistances(utils::NormalizationScheme normScheme) {
            _ihs.normKnnDistances = normScheme;
        };
        void setImageHierarchySettings(const ImageHierarchySettings& ihs) {
            _ihs = ihs;
            setNeighConnection(_ihs.neighborConnection);
            setComponentSim(_ihs.componentSim);
            setNumGeodesicSamples(_ihs.numGeodesicSamples);
            setRandomWalkHandling(_ihs.rwHandling);
            setWeightRandomWalkMergeBySize(_ihs.rwWeightMergeBySize);
        };
        void setRandomWalkSettings(const utils::RandomWalkSettings& rws);
        void setRandomWalkHandling(const utils::RandomWalkHandling& rwh) {
            _ihs.rwHandling = rwh;
        };
        void setWeightRandomWalkMergeBySize(bool rwWeightMergeBySize) {
            _ihs.rwWeightMergeBySize = rwWeightMergeBySize;
        }
        void setGeoCache(const CacheSettings& geoCache) {
            _geoCache = geoCache; 
        }
        void setDataKnnWCC(bool dataKnnWCC) { 
            _dataKnnWCC = dataKnnWCC; 
        }
        void setVerbose(bool verbose) {
            _ihs.verbose = verbose;
        }

    public: // Getter

        std::vector<uint64_t> getSpatialNeighbors(const utils::ComponentID& cid) const;

        const utils::Hierarchy& getHierarchy() const { return _hierarchy; }
        utils::NeighConnection getNeighConnection() const { return _ihs.neighborConnection; }
        const utils::ConnectedNeighbors& getConnectedNeighbors() const { return _neighConnection; }
        utils::ComponentSim getComponentSim() const { return _ihs.componentSim; }
        int64_t getMinimumNumComponents() const { return _ihs.minNumComp; }
        float getMinimumSimilarity() const { return _ihs.maxDist; }
        ImageHierarchySettings& getImageHierarchySettings() { return _ihs; }
        const ImageHierarchySettings& getImageHierarchySettings() const { return _ihs; }
        int64_t getNumRows() const { return _numRows; }
        int64_t getNumCols() const { return _numCols; }
        int64_t getNumDataPoints() const { return _data.getNumPoints(); }
        utils::RandomWalkSettings getNumRandomWalkSettings() const { return _rws; }
        ImageHierarchyStats getStats() const { return _stats; }
        SparseMatSPH& getDataLevelProbdist() { return _dataLevelProbdist; }

        CacheSettings getGeoCachePath() const { return _geoCache; }
        bool getDataKnnWCC() const { return _dataKnnWCC; }
        utils::GraphView& getDataKnnGraphRef() { return _dataKnnGraph; }
        const utils::GraphView& getDataKnnGraph() const { return _dataKnnGraph; }
        bool getVerbose() const { return _ihs.verbose; }

    private:
        void computePreparations();

        void computeBoruvkaHierarchy();

        void connectMostSimilarComponents(uint64_t numComponentsCurrent, uint64_t current_level, utils::BoostGraph& levelGraph);

        utils::Graph computeDistances(uint64_t numComponentsCurrent, uint64_t currentLevel);

        // This is strictly speaking NOT how Boruvka works!
        void mergeAllBelow(float thresh, uint64_t numComponentsCurrent, uint64_t currentLevel, const utils::Graph& distances, utils::BoostGraph& levelGraph, uint64_t& zeroSimCounter, uint64_t& forcedMergeCounter) const;

        // a _ihs.maxDist of -1.f indicates to always merge (if no distance/similarity is available a random neighbor is chosen)
        void mergeMinBelow(float thresh, uint64_t numComponentsCurrent, uint64_t currentLevel, const utils::Graph& distances, utils::BoostGraph& levelGraph, uint64_t& zeroSimCounter, uint64_t& forcedMergeCounter) const;

        void updateHierarchySettings();

    public:
        bool writeStats(const std::string& fileName) const;

    private: // saving and loading
        bool loadCache() override;
        bool writeCache() const override;

        bool checkCacheParameters(const std::string& fileName) const override;
        bool writeCacheParameters(const std::string& fileName) const override;

        bool loadCacheHierarchy(const std::string& fileNameBase);
        bool writeCacheHierarchy(const std::string& fileNameBase) const;

        bool loadCacheImageHierarchy(const std::string& fileNameBase);
        bool writeCacheImageHierarchy(const std::string& fileNameBase) const;

        std::filesystem::path getGeoCacheFile() const;
        bool checkGeoCachePath(const std::filesystem::path& p) const;

    private:

        // Given at instantiation
        utils::GraphView                    _dataKnnGraph = {};
        utils::DataView                     _data = {};
        uint64_t                            _numRows = 0;
        uint64_t                            _numCols = 0;

        // Computed in this class
        utils::Hierarchy                    _hierarchy = {};
        ImageHierarchyStats                 _stats = {};

        SparseMatSPH                        _dataLevelProbdist = {};

        // Settings
        ImageHierarchySettings              _ihs = {};
        utils::ConnectedNeighbors           _neighConnection = {};
        utils::RandomWalkSettings           _rws = {};

        // Caching extras
        CacheSettings                       _geoCache = {};
        bool                                _dataKnnWCC = false;
    };


} // namespace sph

