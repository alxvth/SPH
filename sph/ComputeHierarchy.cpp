#include "ComputeHierarchy.hpp"

#include "sph/utils/Graph.hpp"
#include <sph/utils/Logger.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/ShortestPath.hpp>
#include <sph/utils/Similarities.hpp>

#include <cmath>

namespace sph
{
    ComputeHierarchy::ComputeHierarchy() = default;

    ComputeHierarchy::~ComputeHierarchy() = default;

    void ComputeHierarchy::init(const utils::DataView& data, int64_t rows, int64_t cols, const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
        const utils::RandomWalkSettings& rws, const NearestNeighborsSettings& nns,
        const std::optional<CacheSettings>& cs, const std::optional<CacheSettings>& cs_knn, const std::optional<CacheSettings>& cs_cc, const std::optional<CacheSettings>& cs_geo)
    {
        setSettings(ihs, lss, rws, nns);
        setData(data, rows, cols);
        setCacheSettings(cs, cs_knn, cs_cc, cs_geo);
    }
    
    void ComputeHierarchy::setData(const utils::DataView& data, int64_t rows, int64_t cols)
    {
        _data = data;
        _rows = rows;
        _cols = cols;
    }

    void ComputeHierarchy::setSettings(const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
        const utils::RandomWalkSettings& rws, const NearestNeighborsSettings& nns)
    {
        _nns = nns;
        _ihs = ihs;
        _lss = lss;
        _rws = rws;

        if (_ihs.componentSim == utils::ComponentSim::GEO_CENTROID || _ihs.componentSim == utils::ComponentSim::GEO_WALKS)
            _nns.computeConnectComponents = true;

    }

    void ComputeHierarchy::setCacheSettings(const std::optional<CacheSettings>& cs, const std::optional<CacheSettings>& cs_knn,
        const std::optional<CacheSettings>& cs_cc, const std::optional<CacheSettings>& cs_geo)
    {
        if (cs.has_value())
        {
            _cs = cs.value();
            _hasCacheSettings = true;
        }

        if (cs_knn.has_value())
        {
            _cs_knn = cs_knn.value();
            _hasKnnCacheSettings = true;
        }

        if (cs_cc.has_value())
        {
            _cs_cc = cs_cc.value();
            _hasWccCacheSettings = true;
        }

        if (cs_geo.has_value())
        {
            _cs_geo = cs_geo.value();
            _hasGeoCacheSettings = true;
        }
    }

    void ComputeHierarchy::setKnnDataCacheInfo(const std::string& path, const std::string& fileName, bool cacheActive, bool ignoreSubfolder, const std::string& customSubfolder)
    {
        Log::info("ComputeHierarchy::setKnnDataCacheInfo");
        _knnDataLevel->setCachePathInfo(path, fileName, ignoreSubfolder, customSubfolder);
        _knnDataLevel->setCachingActive(cacheActive);
    }

    void ComputeHierarchy::setImageHierarchyCacheInfo(const std::string& path, const std::string& fileName, bool cacheActive)
    {
        Log::info("ComputeHierarchy::setImageHierarchyCacheInfo");
        _imageHierarchy->setCachingDependency(_knnDataLevel.get());
        _imageHierarchy->setCachePathInfo(path, fileName);
        _imageHierarchy->setCachingActive(cacheActive);
    }

    void ComputeHierarchy::setLevelSimilaritiesCacheInfo(const std::string& path, const std::string& fileName, bool cacheActive)
    {
        Log::info("ComputeHierarchy::setLevelSimilaritiesCacheInfo");
        _levelSimilarities->setCachingDependency(_imageHierarchy.get());
        _levelSimilarities->setCachePathInfo(path, fileName);
        _levelSimilarities->setCachingActive(cacheActive);
    }

    void ComputeHierarchy::computeKnnGraph()
    {
        // 1. Create knn graph on data level
        Log::info("ComputeHierarchy:: Nearest Neighbors on data level");

        // 1.0 Compute graph
        _knnDataLevel = std::make_unique<NearestNeighbors>(_data);

        if (_hasKnnCacheSettings)
            setKnnDataCacheInfo(_cs_knn.path, _cs_knn.fileName, _cs_knn.cacheActive, _cs_knn.ignoreSubfolder, _cs_knn.customSubfolder);
        else if (_hasCacheSettings)
            setKnnDataCacheInfo(_cs.path, _cs.fileName, _cs.cacheActive);

        if (_hasWccCacheSettings)
            _knnDataLevel->setCcCache(_cs_cc);

        _knnDataLevel->compute(_nns);

        utils::Logger::flush();

        // 1.2 Create helper structures
        Log::info("ComputeHierarchy:: Image Hierarchy, with selected NeighConnection {0}, and {1} minimum number of components with {2} data points", (_ihs.neighborConnection == utils::NeighConnection::FOUR) ? "FOUR" : "EIGHT", _ihs.minNumComp, _data.getNumPoints());

        utils::GraphView dataGraph = _knnDataLevel->getKnnGraphView();

        if (!_nns.symmetricNeighbors && !_nns.neighborConnectComponents)
            Log::info("ComputeHierarchy:: Using data knn");

        if (_nns.symmetricNeighbors)
        {
            Log::info("ComputeHierarchy:: Using data knn with symmetrized edges");
            dataGraph = _knnDataLevel->getSymGraphView();
            _knnDataLevel->clearDataGraph();
        }

        if (_nns.neighborConnectComponents)
        {
            Log::info("ComputeHierarchy:: Using data knn with connected components");
            dataGraph = _knnDataLevel->getConnectedGraphView();
            _knnDataLevel->clearSymGraph();
        }

        if (!_knnDataLevel->hasComponentsConnected())
        {
            int64_t numComponents = _knnDataLevel->getNumConnectedComponents();
            Log::info("ComputeHierarchy:: Setting connected components ({})", numComponents);
            _ihs.componentLabels = _knnDataLevel->getConnectedComponents();
            _lss.componentLabels = _knnDataLevel->getConnectedComponents();

            if (_ihs.minNumComp < numComponents)
            {
                Log::warn("ComputeHierarchy:: User-set minimum number of components ({0}) is smaller than number of connected components ({1}). They are adjusted to be equal now ({1}).", _ihs.minNumComp, numComponents);
                _ihs.minNumComp = numComponents;
            }
        }

        _imageHierarchy = std::make_unique<ImageHierarchy>(dataGraph, _data, _rows, _cols, _nns.neighborConnectComponents);
        _levelSimilarities = std::make_unique<LevelSimilarities>(_imageHierarchy->getHierarchy(), dataGraph, _data, _lss);

        _finishedKnn = true;

        utils::Logger::flush();
    }

    void ComputeHierarchy::computeImageHierarchy()
    {
        if (!_finishedKnn)
        {
            Log::warn("ComputeHierarchy:: cannot compute image hierarchy, call computeKnnGraph() first");
            return;
        }
        
        // 2. Build image hierarchy based on knn graph
        Log::info("ComputeHierarchy:: Build image hierarchy based on knn graph");

        if (_hasCacheSettings)
            setImageHierarchyCacheInfo(_cs.path, _cs.fileName, _cs.cacheActive);

        if (_hasGeoCacheSettings)
            _imageHierarchy->setGeoCache(_cs_geo);

        if (_ihs.componentSim == utils::ComponentSim::GEO_CENTROID || _ihs.componentSim == utils::ComponentSim::GEO_WALKS)
        {
            utils::cache::setUseCacheShortestPath(true);
            const float numP = static_cast<float>(_data.getNumPoints());
            size_t estimated_size = static_cast<size_t>(numP * std::pow(numP, 1.5f) * std::log(numP) / std::log(4.f));
            estimated_size = std::clamp(estimated_size, static_cast<size_t>(10'000), static_cast<size_t>(25'000'000));
            utils::cache::reserveCacheShortestPath(estimated_size);
        }

        utils::cache::setActive(false);   // this does not speed up anything, mostly due to the mutex use I think
        if (utils::cache::isActive())
        {
            const float numP = static_cast<float>(_data.getNumPoints());
            size_t estimated_size = static_cast<size_t>(numP * 0.5f * std::log(numP) / std::log(4.f));
            estimated_size = std::clamp(estimated_size, static_cast<size_t>(1'000), static_cast<size_t>(1'000'000));
            utils::cache::reserve(estimated_size);
        }

        _imageHierarchy->compute(_ihs, _rws);

        _finishedHie = true;

        utils::Logger::flush();
    }


    void ComputeHierarchy::computeLevelSimilarities()
    {
        if (!_finishedHie)
        {
            Log::warn("ComputeHierarchy:: cannot compute level similarities, call computeKnnGraph() and computeImageHierarchy() first");
            return;
        }

        _levelSimilarities->resetOutput();
        _levelSimilarities->initOutput();

        if (_skipLevelSimilarities)
            return;

        // 3. Compute knn on each hierarchy level
        Log::info("ComputeHierarchy:: Compute Neighborhood graphs for each hierarchy level");

        if (_hasCacheSettings)
            setLevelSimilaritiesCacheInfo(_cs.path, _cs.fileName, _cs.cacheActive);

        _levelSimilarities->setImageHierarchy(_imageHierarchy.get());
        _levelSimilarities->compute(_lss);

        if (_ihs.componentSim == utils::ComponentSim::GEO_CENTROID || _ihs.componentSim == utils::ComponentSim::GEO_WALKS)
        {
            utils::cache::clearCacheShortestPath();
            utils::cache::setUseCacheShortestPath(false);
        }

        utils::cache::clear();
        utils::cache::setActive(false);

        if (_ihs.componentSim == utils::ComponentSim::GEO_CENTROID || _ihs.componentSim == utils::ComponentSim::GEO_WALKS)
            utils::printShortestPathStatistics(utils::stats::getShortestPathStatistics());

        if (utils::cache::getUseCacheShortestPath())
            utils::printSimilaritiesStatistics(utils::stats::getSimilaritiesStatistics());

        _finishedLev = true;

        utils::Logger::flush();
    }

    void ComputeHierarchy::computeSymmetricProbabilityDistributions(utils::NormalizationScheme method)
    {
        if (!_finishedLev)
        {
            Log::warn("ComputeHierarchy:: cannot compute symmetric probability distributions, call computeLevelSimilarities() or compute() first");
            return;
        }

        _levelSimilarities->symmetrizeOutput(method);

        _finishedPro = true;

        utils::Logger::flush();
    }

    void ComputeHierarchy::compute()
    {
        utils::printSettings(_ihs, _lss, _nns, _rws);

        resetFinished();

        // 1. Create knn graph on data level
        computeKnnGraph();

        // 2. Build image hierarchy based on knn graph
        computeImageHierarchy();

        // 3. Compute knn on each hierarchy level
        computeLevelSimilarities();
    }

} // namespace sph
