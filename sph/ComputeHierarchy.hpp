#pragma once

#include "sph/ImageHierarchy.hpp"
#include "sph/LevelSimilarities.hpp"
#include "sph/NearestNeighbors.hpp"

#include "sph/utils/Data.hpp"
#include "sph/utils/EvalIO.hpp"
#include "sph/utils/Settings.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace sph
{

    class ComputeHierarchy
    {
    public:
        ComputeHierarchy();
        ~ComputeHierarchy();

        // Calls setData, setSettings and setCacheSettings
        void init(const utils::DataView& data, int64_t rows, int64_t cols, const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
            const utils::RandomWalkSettings& rws, const NearestNeighborsSettings& nns, const std::optional<CacheSettings>& cs = std::nullopt,
            const std::optional<CacheSettings>& cs_knn = std::nullopt, const std::optional<CacheSettings>& cs_cc = std::nullopt, const std::optional<CacheSettings>& cs_geo = std::nullopt);

        inline void init(const utils::ImageStack& image, const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
            const utils::RandomWalkSettings& rws, const NearestNeighborsSettings& nns, const std::optional<CacheSettings>& cs = std::nullopt,
            const std::optional<CacheSettings>& cs_knn = std::nullopt, const std::optional<CacheSettings>& cs_cc = std::nullopt, const std::optional<CacheSettings>& cs_geo = std::nullopt)
        {
            init(image.data.getDataView(), image.height, image.width, ihs, lss, rws, nns, cs, cs_knn, cs_cc, cs_geo);
        }

        // Calls computeKnnGraph, computeImageHierarchy and computeLevelSimilarities
        void compute();

        void setData(const utils::DataView& data, int64_t rows, int64_t cols);
        inline void setData(const utils::Data& data, int64_t rows, int64_t cols) { setData(data.getDataView(), rows, cols);  }
        inline void setData(const utils::ImageStack& image) { setData(image.data.getDataView(), image.height, image.width); }

        void setSettings(const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
            const utils::RandomWalkSettings& rws, const NearestNeighborsSettings& nns);
        void setCacheSettings(const std::optional<CacheSettings>& cs = std::nullopt, const std::optional<CacheSettings>& cs_knn = std::nullopt, 
            const std::optional<CacheSettings>& cs_cc = std::nullopt, const std::optional<CacheSettings>& cs_geo = std::nullopt);

        void computeKnnGraph();
        void computeImageHierarchy();
        void computeLevelSimilarities();
        void computeSymmetricProbabilityDistributions(utils::NormalizationScheme method);

        void setSkipLevelSimilarities(bool skipLevelSimilarities) { _skipLevelSimilarities = skipLevelSimilarities; }

    public: // Getter
        const NearestNeighbors* getKnnDataLevel() const { return _knnDataLevel.get(); }
        const ImageHierarchy* getImageHierarchy() const { return _imageHierarchy.get(); }
        const LevelSimilarities* getLevelSimilarities() const { return _levelSimilarities.get(); }

        NearestNeighbors* getKnnDataLevelRef() { return _knnDataLevel.get(); }
        ImageHierarchy* getImageHierarchyRef() { return _imageHierarchy.get(); }
        LevelSimilarities* getLevelSimilaritiesRef() { return _levelSimilarities.get(); }

        bool getSkipLevelSimilarities() const { return _skipLevelSimilarities; }

        NearestNeighborsSettings getNearestNeighborsSettings() const { return _nns; }
        LevelSimilaritiesSettings getLevelSimilaritiesSettings() const { return _lss; }
        ImageHierarchySettings getImageHierarchySettings() const { return _ihs; }
        utils::RandomWalkSettings getRandomWalkSettings() const { return _rws; }

        bool hasKnnGraph() const { return _finishedKnn; }
        bool hasImageHierarchy() const { return _finishedHie; }
        bool hasLevelSimilarities() const { return _finishedLev; }
        bool hasSymmetricProbDist() const { return _finishedPro; }

    private:
        void setKnnDataCacheInfo(const std::string& path, const std::string& fileName, bool cacheActive, bool ignoreSubfolder = false, const std::string& customSubfolder = "");
        void setImageHierarchyCacheInfo(const std::string& path, const std::string& fileName, bool cacheActive);
        void setLevelSimilaritiesCacheInfo(const std::string& path, const std::string& fileName, bool cacheActive);

        inline void resetFinished() {
            _finishedKnn = false;
            _finishedHie = false;
            _finishedLev = false;
            _finishedPro = false;
        }

    private:
        // Compute classes
        std::unique_ptr<NearestNeighbors>   _knnDataLevel = {};
        std::unique_ptr<ImageHierarchy>     _imageHierarchy = {};
        std::unique_ptr<LevelSimilarities>  _levelSimilarities = {};

        // Settings
        NearestNeighborsSettings            _nns = {};
        LevelSimilaritiesSettings           _lss = {};
        ImageHierarchySettings              _ihs = {};
        utils::RandomWalkSettings           _rws = {};
        CacheSettings                       _cs = {};
        CacheSettings                       _cs_knn = {};
        CacheSettings                       _cs_cc = {};
        CacheSettings                       _cs_geo = {};
    
        utils::DataView                     _data = {};
        int64_t                             _rows = {};
        int64_t                             _cols = {};

        bool                                _skipLevelSimilarities = false;
        bool                                _hasCacheSettings = false;
        bool                                _hasKnnCacheSettings = false;
        bool                                _hasWccCacheSettings = false;
        bool                                _hasGeoCacheSettings = false;

        bool                                _finishedKnn = false;
        bool                                _finishedHie = false;
        bool                                _finishedLev = false;
        bool                                _finishedPro = false;
    };

} // namespace sph

