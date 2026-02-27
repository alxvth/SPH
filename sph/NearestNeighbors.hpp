#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "utils/Cacheable.hpp"
#include "utils/Data.hpp"
#include "utils/Graph.hpp"
#include "utils/Settings.hpp"

// TODO: expose knn algorithm parameters in NearestNeighborsSettings

namespace sph {

    /// //////////////// ///
    /// NearestNeighbors ///
    /// //////////////// ///

    class NearestNeighbors : public utils::Cacheable
    {
    public:
        NearestNeighbors();
        NearestNeighbors(const utils::Data& data, const NearestNeighborsSettings& nns = {});
        NearestNeighbors(const utils::DataView& dataView, const NearestNeighborsSettings& nns = {});
        
        ~NearestNeighbors() = default;

        // if k == -1 use the knnGraph.k from constructor
        void compute(const std::optional<NearestNeighborsSettings>& nns = std::nullopt);

        // For symmetric graph, computes labelGraphStrongComponents, otherwise labelGraphWeakComponents
        std::pair<int64_t, std::shared_ptr<std::vector<int64_t>>> computeConnectedComponents();

        // Calls computeConnectedComponents if !hasComponentsComputed
        const utils::GraphView connectComponents();

        const utils::GraphView computeSymmetrizedNnGraph();

        void clearDataGraph() { _knnGraph.clear(); };
        void clearSymGraph() { _symGraph.clear(); };
        void clearConnectedGraph() { _connectedGraph.clear(); };

        void printKnnGraphInfo(const std::string& caller = "");
        void printSymGraphInfo(const std::string& caller = "");
        void printConGraphInfo(const std::string& caller = "");

        inline static utils::KnnIndex indexHeuristic(int64_t numDataPoint) {
            utils::KnnIndex index = utils::KnnIndex::BruteForce;

            if (numDataPoint > 10'000)
                index = utils::KnnIndex::IVFFlat;
            if (numDataPoint > 100'000)
                index = utils::KnnIndex::HNSW;
            if (numDataPoint > 25'000'000)
                index = utils::KnnIndex::HNSWSQ;
            if (numDataPoint > 50'000'000)
                index = utils::KnnIndex::HNSW_IVFPQ;

            return index;
        }

    public: // Setter
        void setData(const utils::Data& data);
        void setData(const utils::DataView& data);

        void setNearestNeighborsSettings(const NearestNeighborsSettings& nns) { _nns = nns; };
        void setMetric(utils::KnnMetric knnMetric);
        void setIndex(utils::KnnIndex knnIndex) { _nns.knnIndex = knnIndex; }

        // The KnnMetric::L2 metric computes squared euclidean distances by default
        // If you set _L2squared false, .compute() will return their sqrt, i.e. proper L2
        void setL2squared(bool b) { _nns.L2squared = b; };

        void setCcCache(const CacheSettings& ccCache) { _ccCache = ccCache; }

    public: // Getter
        std::vector<float>* getKnnDistanceSquared() { return &_knnGraph.knnDistances; };
        std::vector<int64_t>* getKnnIndices() { return &_knnGraph.knnIndices; };

        NearestNeighborsSettings getNearestNeighborsSettings() const { return _nns; };
        utils::KnnMetric getKnnMetric() const { return _nns.knnMetric; };
        utils::KnnIndex getKnnIndex() const { return _nns.knnIndex; };

        utils::DataView getNnData() const { return _faissData; }

        bool getL2squared() const { return _nns.L2squared; };
        bool hasNeighbors() const { return _hasNeighbors; };
        bool hasSymmetric() const { return _hasSymmetric; };
        bool hasComponentsConnected() const { return _hasConnected; }
        bool hasComponentsComputed() const {
            return _numConnectedComponents != -1 && _connectedComponents.get() && _connectedComponents->size() == static_cast<size_t>(_data.getNumPoints());
        }
        const utils::Graph& getDataKnnGraph() const { return _knnGraph; };
        const utils::GraphView getDataKnnGraphView() const { return _knnGraph.getGraphView(); };
        const utils::Graph& getSymGraph() const { return _symGraph; };
        const utils::GraphView getSymGraphView() const { return _symGraph.getGraphView(); };
        const utils::Graph& getConnectedGraph() const { return _connectedGraph; };
        const utils::GraphView getConnectedGraphView() const { return _connectedGraph.getGraphView(); };

        const utils::Graph& getKnnGraph() const
        {
            if (hasSymmetric() && hasComponentsConnected())
                return _connectedGraph;
            else if (hasSymmetric())
                return _symGraph;
            else if (hasComponentsConnected())
                return _connectedGraph;
            else
                return _knnGraph;
        }

        const utils::GraphView getKnnGraphView() const
        {
            if (hasSymmetric() && hasComponentsConnected())
                return _connectedGraph.getGraphView();
            else if (hasSymmetric())
                return _symGraph.getGraphView();
            else if (hasComponentsConnected())
                return _connectedGraph.getGraphView();
            else
                return _knnGraph.getGraphView();
        }

        // returns 0 when not yet computed
        int64_t getNumNearestNeighbors() const { return _nns.numNearestNeighbors; };

        const std::shared_ptr<std::vector<int64_t>> getConnectedComponents() const { return _connectedComponents; }
        std::shared_ptr<std::vector<int64_t>> getConnectedComponentsRef() { return _connectedComponents; }
        int64_t getNumConnectedComponents() const { return _numConnectedComponents; }

        CacheSettings getCcCachePath() const { return _ccCache; }

    private:
        void squareRootOfAllDistances();

        bool checkAllNeighborsExist();
        void ensureFloatingPointIntegrity();
        void checkDistancesAreNonDecreasing(utils::Graph& g);

        void initFaissData();

    private: // saving and loading
        bool loadCache() override;
        bool writeCache() const override;

        bool checkCacheParameters(const std::string& fileName) const override;
        bool writeCacheParameters(const std::string& fileName) const override;

        std::filesystem::path getConnectedComponentsCacheFile(bool symmetric) const;
        bool checkCCCachePath(const std::filesystem::path& p) const;

        std::string fullCachePath() const { return (_cachePath / _cacheFileName).string() + "_nns"; }
        std::string pathParameter() const { return fullCachePath() + "_parametersDataKnn.cache"; }
        std::string pathDataKnn() const { return fullCachePath() + "_DataKnn"; }
        std::string pathSymKnn() const { return fullCachePath() + "_SymKnn"; }
        std::string pathConKnn() const { return fullCachePath() + "_ConKnn"; }
        std::string pathNumComp() const { return fullCachePath() + "_NumComp.cache"; }
        std::string pathConComp() const { return fullCachePath() + "_ConComp.cache"; }

    private:
        using shared_vi64 = std::shared_ptr<std::vector<int64_t>>;

        // Given at instantiation
        utils::DataView             _data = {};

        // Computed in this class
        utils::Graph                _knnGraph = {};
        bool                        _hasNeighbors = false;

        std::vector<float>          _dataNormed = {};                                   // used for KnnMetric::COSINE
        shared_vi64                 _connectedComponents = std::make_shared<vi64>();    // call compConnectedComponents to compute
        int64_t                     _numConnectedComponents = -1;                       // call compConnectedComponents to compute

        utils::Graph                _symGraph = {};                                     // symmetrized _knnGraph
        bool                        _hasSymmetric = false;

        utils::Graph                _connectedGraph = {};                               // connected _knnGraph
        bool                        _hasConnected = false;

        // Settings
        NearestNeighborsSettings    _nns = {};

        // Used for computation
        int64_t                     _faissMetric = 1;                                   // = faiss::METRIC_L2
        utils::DataView             _faissData = {};

        // Caching extras
        CacheSettings               _ccCache = {};
    };


} // namespace sph
