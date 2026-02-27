#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "CommonDefinitions.hpp"
#include "Graph.hpp"
#include "GraphBoost.hpp"
#include "ImageHelper.hpp"
#include "Settings.hpp"

#include <Eigen/SparseCore>

namespace sph::utils
{
    struct ComponentID
    {
        ComponentID() = default;
        ComponentID(uint64_t l, uint64_t i) : 
            level(l), 
            id(i) 
        {}

        uint64_t level = 0;
        uint64_t id = 0;

        inline bool operator==(const ComponentID& rhs) const {
            return level == rhs.level && id == rhs.id;
        }
    };

    /**
     * Hierarchy containing parent-child information for each component
     */
    struct Hierarchy
    {
    public:
        struct Settings {
            ImageInfo imageInfo = {};
            ComponentSim componentSim = ComponentSim::NEIGH_WALKS;

            NormType rwNormSim = NormType::ONEDIM;
            bool rwWeightMergeBySize = true;
            RandomWalkHandling rwHandling = RandomWalkHandling::MERGE_RW_ONLY;
            bool rwRemoveSelfSimAfterMerging = true;    // does not apply to RandomWalkHandling::MERGE_RW_ONLY
            NormalizationScheme normMergedDataDistances = NormalizationScheme::TSNE;

            size_t numGeodesicSamples = std::numeric_limits<size_t>::max();     // a reasonable value might be 100

            bool verbose = false;
        };

        struct AddLevelInfo {
            explicit AddLevelInfo(const RandomWalkSettings& rs, const int64_t ncn, const std::vector<int64_t>& cln) :
                rwsSettings(rs),
                numComponentsNext(ncn),
                componentLabelsNext(&cln)
            { }
            RandomWalkSettings rwsSettings = {};
            int64_t numComponentsNext = 0;
            const std::vector<int64_t>* componentLabelsNext = nullptr;
        };

    public:
        Hierarchy() = default;

    public:
        template<typename intType = uint64_t>
        void getRepresentedDataPoints(const ComponentID& cid, std::vector<intType>& repIDs) const;

        void addSpatialNeighbors(const ComponentID& cid, const vui64& spNeigh);

        auto getNumLevels() const { return numComponents.size(); }

        const vui64& parentsOn(int64_t level) const;

        const vvui64& childrenOn(int64_t level) const;

        const vvui64& mapFromPixelToLevel() const { return pixelComponents; };

        const vvui64& spatialNeighborsOn(int64_t level) const;

        const vui64& pixelComponentsOn(int64_t level) const { return pixelComponents[level]; };

        uint64_t numComponentsOn(int64_t level) const { return numComponents[level]; }

    public:
        void initFirstLevel(int64_t numDataPoints);

        void addLevel(const AddLevelInfo& nextLvlInfo);

        void clear();

        void setSettings(const Settings& s) {
            settings = s;
        }

    private:
        void updateParentsAndChildren(int64_t numComponentsNext, const vi64& componentLabelsNext);
        void updateSpatialNeighbors(const ImageInfo& imageInfo);
        void updateComponentMap();
        void updateRandomWalks(const AddLevelInfo& nextLvlInfo);

    public:
        vui64 numComponents = {};

        // For each level L<L_top (0, 1, 2, ..., L_top-1): 
        // for each component C in L the ID of the component in L+1 that C merges into
        vvui64 parents = {};

        // For each level L>0 (1, 2, ..., L_top): a vector for each component C in L
        // storing the IDs of components in L-1 that merged into C
        std::vector<vvui64> children = {};

        // For each level L>0 (1, 2, ..., L_top): a vector for each component C in L
        // storing the IDs of spatial neighbor components in L
        std::vector<vvui64> spatialNeighbors = {};

        // For each level a vector of size image rows * image cols
        // storing the component ID of the pixel at the level
        //    level  pixel  mapped-to component on level
        vvui64 pixelComponents = {};

        // For each level a map from a component to the pixels that it represents
        // a component might map to multiple pixels, except on bottom
        //    level   component    mapped-to pixels
        std::vector<vvui64> mapFromLevelToPixel = {};

        // For each level a sparse matrix of random walk similarities
        std::vector<std::vector<SparseVecSPH>> randomWalks = {};

        std::vector< std::unique_ptr<GraphInterface>> mergedDataGraphs = {};

        // counts children[i][j].size() == 1
        vvui64 notMergedNodes = {};

        Settings settings = {};

        std::unique_ptr<BoostGraph> bgraph = nullptr;
    };

} // namespace sph::utils
