#pragma once

#include "utils/Cacheable.hpp"
#include "utils/CommonDefinitions.hpp"
#include "utils/Data.hpp"
#include "utils/Graph.hpp"
#include "utils/Hierarchy.hpp"
#include "utils/Settings.hpp"

#include <hdi/data/map_mem_eff.h>   // for vSparseMatHDI

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sph {

    class ImageHierarchy;

    /// ////////////////// ///
    /// Level Similarities ///
    /// ////////////////// ///

    struct LevelSimilaritiesStats
    {
        vf32    perplexities = {};
        vi64    ks = {};
        vf32    avgNumNeighbors = {};
    };

    /**
     * LevelSimilarities class
     * Computes the kNN graph for a Hierarchy on a specific level wrt to a custom similarity measure
     * Or interprets random walk values as probability distribution
     * The computed probability distribution is NOT symmetric since we apply a gaussian norm
     */
    class LevelSimilarities : public utils::Cacheable
    {
    public:
        LevelSimilarities();
        LevelSimilarities(const utils::Hierarchy& hierarchy, const utils::Graph& dataNnGraph, const utils::Data& data, const LevelSimilaritiesSettings& lls);
        LevelSimilarities(const utils::Hierarchy& hierarchy, const utils::GraphView& dataNnGraph, const utils::DataView& data, const LevelSimilaritiesSettings& lls);
        ~LevelSimilarities() = default;

        void compute(const std::optional<LevelSimilaritiesSettings>& lls = std::nullopt);

        void resetOutput();

        void initOutput();

        // 0: do nothing, 1: t-SNE, 2: UMAP
        void symmetrizeOutput(utils::NormalizationScheme method = utils::NormalizationScheme::TSNE);

    public: // Setter
        void setComponentSim(const utils::ComponentSim& sim) { _lss.componentSim = sim; }
        void setKs(const vi64& ks) { _lss.ks = ks; };
        void setForcePopulateSimilaritiesVector(bool force) { _lss.forceComputeDistances = force; };
        void setHierarchy(const utils::Hierarchy* h) { _hierarchy = h; }
        void setDataKnnGraph(const utils::GraphView& graphView) { _dataKnnGraph = graphView; }
        void setDataKnnGraph(const utils::Graph& graph) { setDataKnnGraph(graph.getGraphView()); }
        void setConnectedComponents(std::shared_ptr<vi64> componentLabels) {
            _lss.componentLabels = componentLabels;
        };
        void setLevelSimilaritiesSettings(const LevelSimilaritiesSettings& lls) { _lss = lls; }
        void setImageHierarchy(ImageHierarchy* imgHierarchy) { _imgHierarchy = imgHierarchy; }

    public: // Getter
        const std::vector<float>& getDistances(int64_t level) const { return _distanceGraphs[level].getKnnDistances(); }
        const std::vector<int64_t>& getDistanceIndices(int64_t level) const { return _distanceGraphs[level].getKnnIndices(); }

        const std::vector<float>& getDistancesCurrent() const { return *_knnDistances; }
        const std::vector<int64_t>& getDistancesIndicesCurrent() const { return *_knnIndices; }

        utils::GraphView getSimilaritiesGraphCurrent() const {
            return _distanceGraphs[_currentLevel].getGraphView();
        }

        utils::GraphView getSimilaritiesGraph(int64_t level) const {
            return _distanceGraphs[level].getGraphView();
        }

        const auto& getDistanceGraphsOnAllLevels() const { return _distanceGraphs; }

        const SparseMatHDI& getProbDist(int64_t level) const { return _probDistsAllLevels[level]; }
        SparseMatHDI& getProbDistRef(int64_t level) { return _probDistsAllLevels[level]; }
        const vSparseMatHDI& getProbDistsOnAllLevels() const { return _probDistsAllLevels; }

        auto getCurrentLevel() const { return _currentLevel; }
        auto& getPerplexities() const { return _perplexityOnLevel; }
        auto& getKs() const { return _Ks; }
        auto& getKs(int64_t level) const { return _Ks[level]; }
        auto& getComponentSim() const { return _lss.componentSim; }
        bool getForcePopulateSimilaritiesVector() const { return _lss.forceComputeDistances; }
        LevelSimilaritiesSettings getLevelSimilaritiesSettings() const { return _lss; }
        utils::NormalizationScheme getProbDistIsSymmetric() const { return _probDistIsSymmetric; }

    private:
        // populates _similaritiesVector
        void computeNearestNeighborOnLevel();

        // populates _probDistsAllLevels
        void computeProbDistOnLevel();

    private:
        void updateNumberOfNeighbors();

    public:
        bool writeStats(const std::string& fileName) const;

    private: // saving and loading
        bool loadCache() override;
        bool writeCache() const override;

        bool checkCacheParameters(const std::string& fileName) const override;
        bool writeCacheParameters(const std::string& fileName) const override;

        bool loadCacheSimilarities(const std::string& fileNameBase);
        bool writeCacheSimilarities(const std::string& fileNameBase) const;

        bool loadCacheProbDist(const std::string& fileNameBase);
        bool writeCacheProbDist(const std::string& fileNameBase) const;

        bool loadCacheKs(const std::string& fileNameBase);
        bool writeCacheKs(const std::string& fileNameBase) const;

    private:

        // Given at instantiation
        const utils::Hierarchy*     _hierarchy = nullptr;
        utils::GraphView            _dataKnnGraph = {};
        utils::DataView             _data = {};

        // Given optionally
        ImageHierarchy*             _imgHierarchy = nullptr;

        // Computed in this class
        std::vector<utils::Graph>   _distanceGraphs = {};
        vf32*                       _knnDistances = nullptr;            // will point to _distanceGraphs[_currentLevel].second
        vi64*                       _knnIndices = nullptr;              // will point to _distanceGraphs[_currentLevel].first
        vvi64                       _Ks = {};                           // vectors of number of nearest neighbor per level, driven by _lss.k
        vf32                        _perplexityOnLevel = {};            // perplexity on each level
        vSparseMatHDI               _probDistsAllLevels = {};           // computed from _distanceGraphs or directly taken from hierarchy
        vi64                        _numbersOfPoints = {};              // number of points on level, same as _hierarchy->numComponents but signed

        // Settings
        int64_t                     _currentLevel = 0;
        int64_t                     _currentK = 0;
        LevelSimilaritiesSettings   _lss = {};

        utils::NormalizationScheme  _probDistIsSymmetric = utils::NormalizationScheme::NONE;

        // Stats
        LevelSimilaritiesStats      _stats = {};
    };


} // namespace sph
