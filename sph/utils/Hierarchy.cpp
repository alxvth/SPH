#include "Hierarchy.hpp"

#include "Algorithms.hpp"
#include "Graph.hpp"
#include "GraphNormalization.hpp"
#include "ImageHelper.hpp"
#include "Logger.hpp"
#include "PrintHelper.hpp"
#include "SparseMatrixAlgorithms.hpp"
#include "Statistics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <execution>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <type_traits>

#include <Eigen/SparseCore>
#include <hdi/data/map_mem_eff.h>

namespace sph::utils {

    template<typename intType>
    void Hierarchy::getRepresentedDataPoints(const ComponentID& cid, std::vector<intType>& repIDs) const
    {
        repIDs.clear();

        // If we are on the lowest level, the id only represents itself
        if (cid.level == 0)
        {
            if constexpr (std::is_same_v<intType, uint64_t>)
                repIDs.emplace_back(cid.id);
            else
                repIDs.emplace_back(static_cast<intType>(cid.id));

            return;
        }
        // otherwise we need to go level(s) down
        else
        {
            if constexpr (std::is_same_v<intType, uint64_t>)
                repIDs = children[cid.level - 1][cid.id];
            else
            {
                std::transform(children[cid.level - 1][cid.id].begin(), children[cid.level - 1][cid.id].end(), std::back_inserter(repIDs),
                    [](uint64_t val) { return static_cast<intType>(val); });
            }
        }

        // If we are on the first level, we only need to go done once
        if (cid.level == 1)
            return;

        // If we are on higher level, we only need to expand all represented IDs
        std::vector<intType> idsLower;
        for (uint64_t level = cid.level - 1; level > 0; level--)
        {
            idsLower.clear();
            for (auto& id : repIDs) {
                if constexpr (std::is_same_v<intType, uint64_t>)
                    idsLower.insert(idsLower.end(), children[level - 1][id].begin(), children[level - 1][id].end());
                else
                    std::transform(children[level - 1][id].begin(), children[level - 1][id].end(), std::back_inserter(idsLower),
                        [](uint64_t val) { return static_cast<intType>(val); });
            }

            std::swap(repIDs, idsLower);
        }
    }
    template void Hierarchy::getRepresentedDataPoints<uint64_t>(const ComponentID& cid, std::vector<uint64_t>& repIDs) const;
    template void Hierarchy::getRepresentedDataPoints<uint32_t>(const ComponentID& cid, std::vector<uint32_t>& repIDs) const;
    template void Hierarchy::getRepresentedDataPoints<int64_t>(const ComponentID& cid, std::vector<int64_t>& repIDs) const;
    template void Hierarchy::getRepresentedDataPoints<int32_t>(const ComponentID& cid, std::vector<int32_t>& repIDs) const;

    void Hierarchy::addSpatialNeighbors(const ComponentID& cid, const vui64& spNeighs)
    {
        vui64 newSpNeighs;
        newSpNeighs.reserve(spNeighs.size());

        auto* currentSpNeigh = &(spatialNeighbors[cid.level][cid.id]);

        assert(std::is_sorted(currentSpNeigh->begin(), currentSpNeigh->end()));

        // Determine which spatial neighbors are new
        std::set_difference(
            currentSpNeigh->begin(), currentSpNeigh->end(),
            spNeighs.begin(), spNeighs.end(),
            std::back_inserter(newSpNeighs));

        // Add new spatial neighbors and keep currentSpNeigh sorted
        auto it = currentSpNeigh->begin();
        for (const auto& sn : newSpNeighs) {
            it = std::upper_bound(it, currentSpNeigh->end(), sn);
            currentSpNeigh->insert(it, sn);
        }
    }

    const vui64& Hierarchy::parentsOn(int64_t level) const {
        assert(static_cast<size_t>(level) < getNumLevels());
        return parents[level];
    };

    const vvui64& Hierarchy::childrenOn(int64_t level) const {
        assert(level > 0);
        return children[level - 1];
    };

    const vvui64& Hierarchy::spatialNeighborsOn(int64_t level) const {
        assert(level > 0);
        return spatialNeighbors[level - 1];
    };

    void Hierarchy::initFirstLevel(int64_t numDataPoints)
    {
        assert(numComponents.size() == 0);
        assert(pixelComponents.size() == 0);

        numComponents.push_back(numDataPoints);

        // for level 0 all pixels are individual components
        auto& pixelComponentsFirst = pixelComponents.emplace_back(numDataPoints);
        std::iota(pixelComponentsFirst.begin(), pixelComponentsFirst.end(), 0);

        // for level 0 all pixels map to themselves
        auto& mapLevelToBottomFirst = mapFromLevelToPixel.emplace_back(numDataPoints);
        for (const auto& compID : std::views::iota(0ull, static_cast<uint64_t>(numDataPoints)))
            mapLevelToBottomFirst[compID] = { compID };
    }

    void Hierarchy::addLevel(const AddLevelInfo& nextLvlInfo)
    {
        assert(numComponents.size() > 0);
        assert(numComponents.size() == pixelComponents.size());

        Log::info("Hierarchy::addLevel: update hierarchy components");

        // new parents and children and updated numComponents
        updateParentsAndChildren(nextLvlInfo.numComponentsNext, *nextLvlInfo.componentLabelsNext);

        // new spatial neighbors
        updateSpatialNeighbors(settings.imageInfo);

        // create mapping from pixels to current components on new level and back
        updateComponentMap();

        // do random walks on new level
        if (settings.componentSim == ComponentSim::NEIGH_WALKS ||
            settings.componentSim == ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP ||
            settings.componentSim == ComponentSim::GEO_WALKS)
            updateRandomWalks(nextLvlInfo);

        assert(numComponents.size() == pixelComponents.size());
        assert(spatialNeighbors.size() == children.size());
        assert(parents.size() == children.size());
        assert(notMergedNodes.size() == children.size());
    }

    void Hierarchy::updateParentsAndChildren(int64_t numComponentsNext, const vi64& componentLabelsNext)
    {
        const auto numDataPoints = numComponents.front();
        const auto numComponentsCurrent = numComponents.back();
        const auto current_level = numComponents.size() - 1;
        numComponents.push_back(numComponentsNext);

        auto& childrenNext = children.emplace_back(numComponentsNext);
        auto& parentsNext = parents.emplace_back(numComponentsCurrent);
        auto& pixelIDsNext = pixelComponents.emplace_back(numDataPoints, INT64_MAX);

        ComponentID currentID{};
        std::vector<uint64_t> repDataIDsCurrent;
        for (const auto& compID : std::views::iota(0ull, numComponentsCurrent))
        {
            currentID = { current_level , compID };
            getRepresentedDataPoints(currentID, repDataIDsCurrent);

            std::for_each(SPH_PARALLEL_EXECUTION
                repDataIDsCurrent.cbegin(),
                repDataIDsCurrent.cend(),
                [&pixelIDsNext, &componentLabelsNext, &compID](const uint64_t& id) { 
                    pixelIDsNext[id] = componentLabelsNext[compID]; 
                });

            childrenNext[componentLabelsNext[compID]].push_back(compID);
            parentsNext[compID] = static_cast<uint64_t>(componentLabelsNext[compID]);
        }

        for (auto& childrenIDs : childrenNext) {
            sortAndUnique(childrenIDs);
        }

        auto& notMergedNodesNext = notMergedNodes.emplace_back();
        for (uint64_t i = 0; i < childrenNext.size(); i++) {
            if (childrenNext[i].size() == 1) {
                notMergedNodesNext.push_back(i);
            }
        }

    }

    void Hierarchy::updateSpatialNeighbors(const ImageInfo& imageInfo)
    {
        const auto numComponentsNext = numComponents.back();
        auto& spatialNeighborsNext = spatialNeighbors.emplace_back(numComponentsNext);
        auto& pixelIDsNext = pixelComponents.back();
        const auto numDataPoints = numComponents.front();

        std::vector<uint64_t> repDataIDsCurrent;
        for (uint64_t pixelId = 0; pixelId < numDataPoints; pixelId++)
        {
            pixelNeighborIDs(imageInfo.numCols, imageInfo.numRows, imageInfo.neighConnection, pixelId, repDataIDsCurrent);

            for (const auto& id : repDataIDsCurrent)
            {
                if (pixelIDsNext[id] != pixelIDsNext[pixelId])
                    spatialNeighborsNext[pixelIDsNext[pixelId]].push_back(pixelIDsNext[id]);
            }
        }

        for (auto& spatialNeighborVec : spatialNeighborsNext) {
            sortAndUnique(spatialNeighborVec);
        }

    }

    void Hierarchy::updateComponentMap()
    {
        assert(numComponents.size() > 1);
        assert(mapFromLevelToPixel.size() == numComponents.size() - 1);

        auto& mapLevelNextToBottom = mapFromLevelToPixel.emplace_back();

        const auto numComponentsCurrent = numComponents.back();
        const auto newLevel = numComponents.size() - 1;

        mapLevelNextToBottom.resize(numComponentsCurrent);

        // Map each ID on newLevel to the bottom components they represent
        ComponentID currentID{};
        for (const auto& compID : std::views::iota(0ull, numComponentsCurrent))
        {
            currentID = { newLevel , compID };
            getRepresentedDataPoints(currentID, mapLevelNextToBottom[compID]);
        }
    }

    void Hierarchy::updateRandomWalks(const AddLevelInfo& nextLvlInfo)
    {
        assert(numComponents.size() > 1);
        assert(parents.size() == numComponents.size() - 1);
        assert(randomWalks.size() == numComponents.size() - 1);

        Log::info("Hierarchy::updateRandomWalks: starting...");

        const auto numComponentsNext = numComponents.back();
        const auto& nodeParents = parents.back();

        std::vector<SparseVecSPH> randomWalksSimilaritiesInput = {};

        // STEP 1: Merge nodes 
        switch (settings.rwHandling)
        {
        case RandomWalkHandling::MERGE_RW_ONLY: [[fallthrough]];
        case RandomWalkHandling::MERGE_RW_NEW_WALKS: [[fallthrough]];
        case RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN:
        {
            // merge random walk nodes
            Log::info("Hierarchy::updateRandomWalks: Merge current random walk similarities");

            assert(mergedDataGraphs.size() == 0);

            const auto& randomWalksCurrent = randomWalks.back();

            if (settings.verbose)
            {
                Log::info("Hierarchy::updateRandomWalks: current level random walks");
                printSparseMatrixAsDense(randomWalksCurrent, true);
            }

            const bool parallelMerge = false; // parallel implementation does not seem to be very efficient;
            randomWalksSimilaritiesInput = mergeNodesRandomWalks(randomWalksCurrent, numComponentsNext, nodeParents, /*norm*/ false, /*weightBySize*/ settings.rwWeightMergeBySize, /*parallel*/ parallelMerge);

            // remove self-walks
            if (settings.rwRemoveSelfSimAfterMerging && randomWalksSimilaritiesInput.size() > 1) {

                if (settings.rwHandling != RandomWalkHandling::MERGE_RW_ONLY)
                    removeDiagonalElements(randomWalksSimilaritiesInput);
                else
                    Log::warn("Hierarchy::updateRandomWalks: Setting RandomWalkHandling::MERGE_RW_ONLY "
                              "will ignore rwRemoveSelfSimAfterMerging and NOT set self-sim to zero as this "
                              "is intended for NEW random walks to encourage further walks.");
            }

            // normalize merged walks
            switch (settings.rwNormSim) {
            case NormType::ONEDIM:
                for (auto& randomWalks : randomWalksSimilaritiesInput)
                    normalizeSparseVector(randomWalks);
                break;

            case NormType::TWODIM:
                normalizeSparseMatrix(randomWalksSimilaritiesInput);
                break;
            }

            break;
        }
        case RandomWalkHandling::MERGE_DATA_NEW_WALKS:
        {
            // merge data knn nodes
            Log::info("Hierarchy::updateRandomWalks: Merge knn nodes");

            assert(mergedDataGraphs.size() == randomWalks.size());
            
            const auto mergedDataGraphCurrent = mergedDataGraphs.back().get();
            Graph mergedGraph = mergeGraphNodes(*mergedDataGraphCurrent, numComponentsNext, nodeParents);

            if (settings.verbose)
            {
                Log::info("Hierarchy: current level data graph connections");
                printGraphAsDenseMatrix(*mergedDataGraphCurrent);
                Log::info("Hierarchy: merged level data graph connections");
                printGraphAsDenseMatrix(mergedGraph);
            }

            // create probability distribution
            Log::info("Hierarchy::updateRandomWalks: Create random walks similarities from merged knn nodes");

            normalizeKnnDistances(mergedGraph, settings.normMergedDataDistances, randomWalksSimilaritiesInput);

            // save merged nodes
            mergedDataGraphs.emplace_back(std::make_unique<Graph>(std::move(mergedGraph)));

            break;
        }
        default:
            Log::error("Hierarchy::updateRandomWalks: unhandled settings.rwHandling case");
            break;
        }

        const auto randomWalksMergedStats = sparseMatrixStats(randomWalksSimilaritiesInput);
        printSparseMatrixStats(randomWalksMergedStats, "Merged random walks");

        if (settings.verbose)
        {
            Log::info("Hierarchy: merged basis for random walks");
            printSparseMatrixAsDense(randomWalksSimilaritiesInput, true);
        }

        std::vector<SparseVecSPH>* randomWalksOutput = nullptr;

        // STEP 2: new random walks
        switch (settings.rwHandling)
        {
        case RandomWalkHandling::MERGE_RW_ONLY:
        {
            Log::info("Hierarchy::updateRandomWalks: No new random walks, use merged random walks");
            randomWalksOutput = &randomWalks.emplace_back(randomWalksSimilaritiesInput);
            break;
        }
        case RandomWalkHandling::MERGE_DATA_NEW_WALKS:  [[fallthrough]];
        case RandomWalkHandling::MERGE_RW_NEW_WALKS:    [[fallthrough]];
        case RandomWalkHandling::MERGE_RW_NEW_WALKS_AND_KNN:
        {
            // new random walks with the merged random walk probabilities
            Log::info("Hierarchy::updateRandomWalks: New random walks");

            randomWalksOutput = &randomWalks.emplace_back();

            if(nextLvlInfo.rwsSettings.normalize && nextLvlInfo.rwsSettings.removeDiagonal)
                Log::info("Hierarchy::updateRandomWalks: (remove diagonal elements of random walks and normalize)");

            SparseMatrixStats rwsStats = {};
            doRandomWalks(randomWalksSimilaritiesInput, nextLvlInfo.rwsSettings, *randomWalksOutput, rwsStats, settings.verbose);

            break;
        }
        default:
            Log::error("Hierarchy::updateRandomWalks: unhandled settings.rwHandling case");
            break;
        }

        // if the top level is only one node, the self-sim should be preserved
        if (randomWalksOutput->size() == 1 && randomWalksOutput->at(0).nonZeros() == 0)
            randomWalksOutput->at(0).coeffRef(0) = 1;

    }

    void Hierarchy::clear() {
        numComponents.clear();
        parents.clear();
        children.clear();
        spatialNeighbors.clear();
        pixelComponents.clear();
        mapFromLevelToPixel.clear();
        randomWalks.clear();
        mergedDataGraphs.clear();
        notMergedNodes.clear();
        bgraph.reset();
    }


} // namespace sph::utils
