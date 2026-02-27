#include "SparseMatrixAlgorithms.hpp"

#include "Algorithms.hpp"
#include "CommonDefinitions.hpp"
#include "Distances.hpp"
#include "Graph.hpp"
#include "GraphUtils.hpp"
#include "Logger.hpp"
#include "MaxSizeDeque.hpp"
#include "PrintHelper.hpp"
#include "ProgressBar.hpp"
#include "Statistics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <execution>
#include <iterator>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>

#ifdef SPH_RELEASE 
#include <omp.h>
#endif

#include <ankerl/unordered_dense.h>
#include <hdi/data/map_mem_eff.h>
#include <range/v3/view/zip.hpp>

namespace sph::utils {

    void doRandomWalks(const SparseMatSPH& similarities, const RandomWalkSettings& settings, SparseMatSPH& randomWalks, SparseMatrixStats& stats, bool verbose)
    {
        using hash = ankerl::unordered_dense::hash<int64_t>;
        using hashset = ankerl::unordered_dense::set<int64_t, hash>;

        Log::info("Random walks: {0} walks with {1} steps each using {2} weighting", settings.numRandomWalks, settings.singleWalkLength, settings.importanceWeighting);

        float pruneValue = settings.pruneValue;
        if (settings.pruneSteps > 0)
        {
            uint64_t pruneStep = std::min(settings.singleWalkLength - 1, settings.pruneSteps);
            uint64_t pruneStepInv = settings.singleWalkLength - pruneStep;

            if (settings.importanceWeighting == ImportanceWeighting::LINEAR)
                pruneValue = stepLinear(pruneStepInv, settings.singleWalkLength);
            else if (settings.importanceWeighting == ImportanceWeighting::NORMAL)
                pruneValue = stepNormal(pruneStepInv, settings.singleWalkLength);
            else if (settings.importanceWeighting == ImportanceWeighting::CONSTANT)
                pruneValue = static_cast<float>(pruneStepInv);
            else if (settings.importanceWeighting == ImportanceWeighting::FIRST_VISIT)
                pruneValue = static_cast<float>(pruneStep) / settings.singleWalkLength;

            if (pruneValue > 0.5f)
            {
                Log::warn("doRandomWalks: current settings would lead to a large prune value of {}. Clamping it to 0.5", pruneValue);
                pruneValue = 0.5f;
            }

        }

        if(pruneValue > 0.f)
            Log::info("Random walks: pruning all values below {}", pruneValue);

        const uint64_t expectedSize = settings.singleWalkLength * 2;
        const uint64_t randomSeed   = settings.randomSeed;
        const auto nd               = similarities.size();

        // square matrix
        randomWalks.clear();
        randomWalks.resize(nd);
        for (auto& sparseVec : randomWalks)
        {
            sparseVec.resize(nd);
            sparseVec.reserve(expectedSize);
        }

        const int numOmpThreads = settings.parallel ? omp_get_max_threads() : 1;

        std::vector<std::mt19937_64> generators(numOmpThreads);
        std::vector<std::uniform_real_distribution<double>> distributions(numOmpThreads, std::uniform_real_distribution<double>(0.0, 1.0));

        std::seed_seq seeds{ randomSeed };
        for (int i = 0; i < numOmpThreads; ++i) {
            generators[i] = std::mt19937_64(seeds);
        }

        uint64_t numElements = 0;
        uint64_t numElementsEffective = 0;
        stats.pruneVale = pruneValue;

        ProgressBar progress(nd);

        // for each data point...
        SPH_PARALLEL_THREADS(numOmpThreads)
        for (int64_t startingPoint = 0; startingPoint < static_cast<int64_t>(nd); ++startingPoint)
        {
            auto& neighborHits  = randomWalks[startingPoint];
            const int ompTreadID = omp_get_thread_num();

            sph::SparseVecSPH hitCount(nd);

#if SPH_DEBUG
            double sum = 0;
#endif

            // ...start a number of random walks
            for (uint64_t walkNum = 0; walkNum < settings.numRandomWalks; ++walkNum)
            {
                float stepWeight        = 1.f;
                auto currentID          = startingPoint;
                double randomThresh     = 0.;
                auto nextID             = currentID;
                double incremental_prob = 0.;

                hashset wasVisited;
                wasVisited.reserve(settings.singleWalkLength);
                wasVisited.insert(currentID);

                // .. and store the visited neighbor at each step
                for (uint64_t currentStep = 0; currentStep < settings.singleWalkLength; ++currentStep)
                {
                    randomThresh        = distributions[ompTreadID](generators[ompTreadID]);
                    nextID              = currentID;
                    incremental_prob    = 0.;

                    for (SparseVecSPH::InnerIterator it(similarities[currentID]); it; ++it) {
                        incremental_prob += it.value();
                        if (randomThresh < incremental_prob) {
                            nextID = it.row();
                            break;
                        }
                    }

                    switch (settings.importanceWeighting)
                    {
                    case ImportanceWeighting::LINEAR: stepWeight = stepLinear(currentStep, settings.singleWalkLength); break;
                    case ImportanceWeighting::NORMAL: stepWeight = stepNormal(currentStep, settings.singleWalkLength); break;
                    case ImportanceWeighting::ONLYLAST: 
                    {
                        if (currentStep < settings.singleWalkLength - 1)
                            stepWeight = 0.f;
                        else
                            stepWeight = 1.f;

                        break;
                    }
                    case ImportanceWeighting::CONSTANT: break;  // nothing to do
                    case ImportanceWeighting::FIRST_VISIT:
                        if (wasVisited.contains(nextID))
                            stepWeight = 0;
                        else
                        {
                            wasVisited.insert(nextID);
                            hitCount.coeffRef(nextID)++;
                            stepWeight = static_cast<float>(currentStep + 1);
                        }
                        break;
                    }

                    if (stepWeight > 0.f)
                    {
                        neighborHits.coeffRef(nextID) += stepWeight;
#if SPH_DEBUG
                        sum += stepWeight;
#endif
                    }

                    currentID = nextID;
                }
            } // walks for startingPoint

            if (settings.importanceWeighting == ImportanceWeighting::FIRST_VISIT)
            {
                assert(hitCount.nonZeros() == neighborHits.nonZeros());

                // Normalize to get average step distance
                for (SparseVecSPH::InnerIterator it(neighborHits); it; ++it)
                {
                    assert(hitCount.coeff(it.index()) > 0);
                    it.valueRef() /= hitCount.coeff(it.index());
                }

                const float m = -1.f / (settings.singleWalkLength - 1.f);
                const float c = settings.singleWalkLength / (settings.singleWalkLength - 1.f);

                // Invert to align distance to other similarities, map from [1, singleWalkLength] to [1, 0] with y = mx + c
                for (SparseVecSPH::InnerIterator it(neighborHits); it; ++it)
                {
                    assert(it.value() >= 1.f);
                    assert(it.value() <= static_cast<float>(settings.singleWalkLength));

                    // prevent some numerical issues where the result could be just below 0;
                    it.valueRef() = std::max(0.f, m * it.value() + c);

                    assert(it.value() <= 1.f);
                    assert(it.value() >= 0.f);
                }
            }

#if SPH_DEBUG
            {
                const auto total_hits   = settings.singleWalkLength * settings.numRandomWalks;
                constexpr double eps    = 0.001f;

                if (settings.importanceWeighting == ImportanceWeighting::CONSTANT)
                    assert(sum == total_hits);
                if (settings.importanceWeighting == ImportanceWeighting::LINEAR)
                    assert(std::abs(sum - settings.numRandomWalks * static_cast<double>(settings.singleWalkLength + 1) / 2) < eps);
                if (settings.importanceWeighting == ImportanceWeighting::ONLYLAST)
                    assert(sum == settings.numRandomWalks);
                // no closed form for the normal weight sum
                // no closed form for the first visit weighting
            }
#endif // SPH_DEBUG

            const uint64_t numElementsNonZeros = neighborHits.nonZeros();

            if (pruneValue > 0.f)
            {
                for (SparseVecSPH::InnerIterator it(neighborHits); it; ++it) {
                    if (it.valueRef() <= pruneValue) {
                        it.valueRef() = 0.f;
                    }
                }
                // Remove the zeros
                const auto numPrunedValues = neighborHits.prune(0.f, Eigen::NumTraits<float>::dummy_precision());
                Log::debug("doRandomWalks: pruned {} values", numPrunedValues);

                neighborHits.data().squeeze();
            }

            const uint64_t numElementsEffectiveNonZeros = neighborHits.nonZeros();

            if (numElementsEffectiveNonZeros == 0)
                Log::warn("doRandomWalks: 0 numElementsEffectiveNonZeros for starting point {0}", startingPoint);

            SPH_PARALLEL_CRITICAL
            {
                numElements += numElementsNonZeros;
                numElementsEffective += numElementsEffectiveNonZeros;

                progress.update();
            }
        }
        progress.finish();

        if (settings.removeDiagonal)
            [[maybe_unused]] uint64_t numRemovedElements = utils::removeDiagonalElements(randomWalks);

        if (settings.normalize)
        {
            SPH_PARALLEL
            for (int64_t p = 0; p < static_cast<int64_t>(nd); ++p)
            {
                utils::normalizeSparseVector(randomWalks[p]);
#if SPH_DEBUG
                {
                    const auto sum = randomWalks[p].sum();
                    assert(std::abs(sum - 1.f) < 0.001f);
                }
#endif // SPH_DEBUG

            }
        }

        if (numElements != numElementsEffective)
            Log::info("Random walks: pruned {} values", numElements - numElementsEffective);

        stats.nonZeros          = numElements;
        stats.totalEntries      = nd * nd;
        if(numElements == numElementsEffective)
            stats.averageNonZeros   = static_cast<double>(numElements) / nd;
        else
            stats.averageNonZeros = static_cast<double>(numElementsEffective) / nd;
        stats.sparsity          = 1.0 - static_cast<double>(numElements)          / (nd * nd);
        stats.sparsityEffective = 1.0 - static_cast<double>(numElementsEffective) / (nd * nd);

        printSparseMatrixStats(stats, "Random walks");

        if (verbose)
        {
            Log::info("Random walks: input similarities");
            printSparseMatrixAsDense(similarities, true);
            Log::info("Random walks: output randomWalks");
            printSparseMatrixAsDense(randomWalks, true);
        }
    }

    SparseMatSPH mergeNodesRandomWalks(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents, bool norm /*true*/, bool weightBySize /*true*/, bool parallel /*true*/)
    {
        SparseMatSPH matrixMerged;

        if (parallel)
            matrixMerged = mergeNodesRandomWalksMultiThread(matrixCurrent, numComponentsMerged, parents, norm, weightBySize);
        else
            matrixMerged = mergeNodesRandomWalksSingleThread(matrixCurrent, numComponentsMerged, parents, norm, weightBySize);

        return matrixMerged;
    }

    SparseMatSPH mergeNodesRandomWalksSingleThread(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents, bool norm /*true*/, bool weightBySize /*true*/)
    {
        Log::info("mergeNodesRandomWalks (SingleThread): merge random walks with weightBySize {0}", weightBySize ? "ON" : "OFF");

        SparseMatSPH matrixMerged(numComponentsMerged);

        auto numComponentsCurrent = parents.size();
        const auto reserveEstimate = matrixCurrent[0].nonZeros();

        for (auto& sparseRow : matrixMerged)
        {
            sparseRow.resize(numComponentsMerged);
            sparseRow.reserve(reserveEstimate);
        }

        utils::ProgressBar progress(numComponentsCurrent);

        std::vector<uint64_t> rowWeights(numComponentsMerged, 0);

        // for each current node...
        for (size_t compID = 0; compID < numComponentsCurrent; compID++)
        {
            auto& fromRow = matrixCurrent[compID];
            const auto mergeIntoID = parents[compID];
            auto& mergeIntoRow = matrixMerged[mergeIntoID];

            const auto rowWeight = weightBySize ? fromRow.nonZeros() : 1ll;
            rowWeights[mergeIntoID] += rowWeight;

            // ... go through its children and add their similarity values to the parent node (weighted by component size)
            for (SparseVecSPH::InnerIterator it(fromRow); it; ++it)
            {
                const auto& mergeIntoColumn = parents[it.index()];
                mergeIntoRow.coeffRef(mergeIntoColumn) += it.value() * rowWeight;
            }

            progress.update();
        }
        progress.finish();

        // normalize after weighting with component size
        if (weightBySize)
            normalizeSparseMatrixWith(matrixMerged, rowWeights);

        if (norm)
            normalizeUnitSparseMatrix(matrixMerged);

        return matrixMerged;
    }

    SparseMatSPH mergeNodesRandomWalksMultiThread(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents, bool norm /*true*/, bool weightBySize /*true*/)
    {
        // TODO: does it make sense to weight by size of random walks?
        Log::info("mergeNodesRandomWalks (MultiThread): merge random walks with weightBySize {0}", weightBySize ? "ON" : "OFF");

        SparseMatSPH matrixMergedTemp(numComponentsMerged);
        SparseMatSPH matrixMerged(numComponentsMerged);

        const size_t numComponentsCurrent   = parents.size();
        const Eigen::Index reserveEstimate  = 2 * matrixCurrent[0].nonZeros();

        for (SparseVecSPH& sparseRow : matrixMergedTemp)
        {
            sparseRow.resize(matrixCurrent.size());
            sparseRow.reserve(reserveEstimate);
        }

        for (SparseVecSPH& sparseRow : matrixMerged)
        {
            sparseRow.resize(numComponentsMerged);
            sparseRow.reserve(reserveEstimate);
        }

        utils::ProgressBar progress(numComponentsCurrent + numComponentsMerged);
        std::vector<uint64_t> rowWeights(numComponentsMerged, 0);

#ifdef SPH_RELEASE 
        std::vector<omp_lock_t> omp_locks(numComponentsCurrent);

        for (omp_lock_t& omp_lock : omp_locks)
            omp_init_lock(&omp_lock);
#endif
        // Merge rows
        SPH_PARALLEL
        for (size_t compID = 0; compID < numComponentsCurrent; compID++)
        {
            const uint64_t mergeIntoID          = parents[compID];
            const SparseVecSPH& mergeFromRow    = matrixCurrent[compID];
            const Eigen::Index rowWeight        = weightBySize ? mergeFromRow.nonZeros() : 1ll;

#ifdef SPH_RELEASE 
            omp_lock_t* current_lock = &omp_locks[mergeIntoID];
            omp_set_lock(current_lock);
#endif

            rowWeights[mergeIntoID]     += rowWeight;
            SparseVecSPH& mergeIntoRow  = matrixMergedTemp[mergeIntoID];
            mergeIntoRow                += (matrixCurrent[compID] * rowWeight);

#ifdef SPH_RELEASE 
            omp_unset_lock(current_lock);
#endif

            SPH_PARALLEL_CRITICAL
            progress.update();
        }

#ifdef SPH_RELEASE 
        for (omp_lock_t& omp_lock : omp_locks)
            omp_destroy_lock(&omp_lock);
#endif

        // Merge columns
        for (size_t compID = 0; compID < numComponentsMerged; compID++)
        {
            SparseVecSPH& mergeFromRow = matrixMergedTemp[compID];
            SparseVecSPH& mergeIntoRow = matrixMerged[compID];
            for (SparseVecSPH::InnerIterator it(mergeFromRow); it; ++it)
            {
                const auto& mergeIntoColumn = parents[it.index()];
                mergeIntoRow.coeffRef(mergeIntoColumn) += it.value();
            }

            mergeFromRow.resize(0);

            progress.update();
        }
        progress.finish();

        // normalize after weighting with component size
        if (weightBySize)
            normalizeSparseMatrixWith(matrixMerged, rowWeights);

        if (norm)
            normalizeUnitSparseMatrix(matrixMerged);

        return matrixMerged;
    }

    SparseMatSPH mergeNodesDataDistances(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents)
    {
        SparseMatSPH matrixMerged(numComponentsMerged);

        auto numComponentsCurrent = parents.size();
        const auto reserveEstimate = matrixCurrent[0].nonZeros();

        for (auto& sparseRow : matrixMerged)
        {
            sparseRow.resize(numComponentsMerged);
            sparseRow.reserve(reserveEstimate);
        }

        utils::ProgressBar progress(numComponentsCurrent);

        // for each current node...
        for (const auto& compID : std::views::iota(0ull, numComponentsCurrent))
        {
            auto& fromRow = matrixCurrent[compID];
            const auto mergeIntoID = parents[compID];
            auto& mergeIntoRow = matrixMerged[mergeIntoID];

            // ... go through its children and add their similarity values to the parent node (weighted by component size)
            for (SparseVecSPH::InnerIterator it(fromRow); it; ++it)
            {
                const auto& id = it.index();
                const auto& val = it.value();
                const auto& mergeIntoColumn = parents[id];

                // Only use if new or smaller distance
                if (mergeIntoRow.coeff(mergeIntoColumn) == 0)
                    mergeIntoRow.coeffRef(mergeIntoColumn) = val;
                else if (mergeIntoRow.coeff(mergeIntoColumn) > val)
                    mergeIntoRow.coeffRef(mergeIntoColumn) = val;
            }

            progress.update();
        }
        progress.finish();

        return matrixMerged;
    }

    Graph mergeGraphNodes(const GraphInterface& graph, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents)
    {
        using hash = ankerl::unordered_dense::hash<int64_t>;
        using hashmap = ankerl::unordered_dense::map<int64_t, float, hash>;

        std::vector<hashmap> mergedNodes(numComponentsMerged);
        const auto numComponentsCurrent = parents.size();

        utils::ProgressBar progress(numComponentsCurrent + numComponentsMerged);

        // STEP 1: merge nodes
        // for each current node...
        for (const auto& compID : std::views::iota(0ull, numComponentsCurrent))
        {
            const auto& fromIndices = graph.getNeighbors(compID);
            const auto& fromDistances = graph.getDistances(compID);
            const auto mergeIntoID = parents[compID];
            auto& mergeIntoRow = mergedNodes[mergeIntoID];

            assert(fromIndices.size() == fromDistances.size());

            // ... go through its children and add their similarity values to the parent node (weighted by component size)
            for (const auto& [id, dist] : ranges::views::zip(fromIndices, fromDistances))
            {
                const auto& mergeIntoColumn = parents[id];

                // Only use if new or smaller distance
                if (!mergeIntoRow.contains(mergeIntoColumn))
                    mergeIntoRow[mergeIntoColumn] = dist;
                else if (mergeIntoRow[mergeIntoColumn] > dist)
                    mergeIntoRow[mergeIntoColumn] = dist;
            }

            progress.update();
        }

        // STEP 2: convert back into graph
        Graph mergedGraph;
        mergedGraph.numPoints = numComponentsMerged;

        for (const auto& compID : std::views::iota(0ull, numComponentsMerged))
        {
            const auto& currentNodePairs = mergedNodes[compID];

            // Sort the neighbors-distance pairs
            std::vector<std::pair<int64_t, float>> pairs(currentNodePairs.cbegin(), currentNodePairs.cend());
            std::sort(SPH_PARALLEL_EXECUTION
                pairs.begin(), 
                pairs.end(), 
                [](const auto& a, const auto& b) {
                if (a.second == b.second)
                    return a.first < b.first;
                return a.second < b.second;
                });

            // Append the neighbors-distance pairs to the graph structure
            for (const auto& pair : pairs) {
                mergedGraph.getKnnIndices().push_back(pair.first);
                mergedGraph.getKnnDistances().push_back(pair.second);
            }

            mergedGraph.getNns().push_back(pairs.size());

            progress.update();
        }
        progress.finish();

        mergedGraph.updateOffsets();

        ensureClosestPointIsSelf(&mergedGraph);

        assert(mergedGraph.isValid());

        return mergedGraph;

    }

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> createSparseMatrixFromGraph(const GraphBaseInterface& graphView, bool verbose)
    {
        if (verbose)
            Log::info("createSparseMatrixFromGraph");

        const size_t rows = graphView.getNumPoints();

        // Calculate average non-zeros per row
        size_t nonZeros = 0;
        for (size_t row = 0; row < rows; ++row)
            nonZeros += graphView.getK(row);

        // Create a vector of triplets
        std::vector<Eigen::Triplet<float, SparseVecSPH::Index>> triplets;
        triplets.reserve(nonZeros);

        for (size_t row = 0; row < rows; ++row) {
            for (size_t k = 1; k < static_cast<size_t>(graphView.getK(row)); k++) {
                const auto& [id, val] = graphView.getNeighborDistanceN(row, k);
                triplets.emplace_back(static_cast<SparseVecSPH::Index>(row), id, val);
            }
        }

        // Create the sparse matrix from the combined triplets
        Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatrix(rows, rows);
        sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
        sparseMatrix.makeCompressed();

        return sparseMatrix;
    }

    void convertGraphToEigenSparse(const GraphBaseInterface& graphView, SparseMatSPH& convertedGraph)
    {
        const auto np = graphView.getNumPoints();

        convertedGraph.clear();
        convertedGraph.resize(np);

        SPH_PARALLEL
        for (int64_t p = 0; p < np; ++p) {
            const auto k = graphView.getK(p);
            convertedGraph[p].resize(np);
            convertedGraph[p].reserve(k);

            for (int64_t n = 1; n < k; ++n)
            {
                const auto dp = graphView.getNeighborDistanceN(p, n);
                convertedGraph[p].coeffRef(static_cast<uint32_t>(dp.first)) = dp.second;
            }

        }

    }

    void normalizeSparseMatrix(SparseMatSPH& sparseMat)
    {
        float normVal = 0;

        for (const auto& sparseVec : sparseMat)
            normVal += sparseVec.sum();

        for (auto& sparseVec : sparseMat)
            for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
                it.valueRef() /= normVal;

    }

    void normalizeSparseMatrixWith(SparseMatSPH& sparseMat, std::vector<uint64_t> rowWeights)
    {
        assert(sparseMat.size() == rowWeights.size());

        int64_t size = static_cast<int64_t>(sparseMat.size());

        SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(size); ++i)
            normalizeSparseVectorWith(sparseMat[i], rowWeights[i]);

    }

    void normalizeMinMaxSparseVector(SparseVecSPH& sparseVec)
    {
        float minValue(1), maxValue(0);
        for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
        {
            if (it.value() > maxValue) {
                maxValue = it.value();
            }
            if (it.value() < minValue) {
                minValue = it.value();
            }
        }

        float range = maxValue - minValue;

        assert(range >= 0);

        if (range == 0)
            range = 1;

        for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
            it.valueRef() = (it.value() - minValue) / range;
    }

    void normalizeUnitSparseVector(SparseVecSPH& sparseVec)
    {
        const auto normVal = sparseVec.sum();

        for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
            it.valueRef() /= normVal;

    }

    void normalizeUnitSparseMatrix(SparseMatSPH& sparseMat, bool verbose)
    {
        utils::ProgressBar progress(sparseMat.size(), verbose);

        for (auto& sparseRow : sparseMat)
        {
            // ... normalize each row
            normalizeUnitSparseVector(sparseRow);

#if SPH_DEBUG
            {
                float normSum = sparseRow.sum();
                assert(std::abs(normSum - 1.f) < 0.0001f);
            }
#endif // SPH_DEBUG

            progress.update();
        }
        progress.finish();

    }

    void removeElement(SparseVecSPH& sparseVec, size_t elem)
    {
        sparseVec.coeffRef(elem) = 0.0f;
        sparseVec.prune(0.f, Eigen::NumTraits<float>::dummy_precision());
        sparseVec.data().squeeze();
    }

    uint64_t removeDiagonalElements(SparseMatSPH& sparseMat, bool keepSingleEntry /* = true */)
    {
        uint64_t numRemovedElements = 0;

        for (uint64_t i = 0; i < sparseMat.size(); i++)
        {
            if (keepSingleEntry && sparseMat[i].nonZeros() <= 1)
                continue;

            removeElement(sparseMat[i], i);
            numRemovedElements++;
        }

        return numRemovedElements;
    }
    
    std::vector<std::pair<uint32_t, float>> findTopK(const SparseVecSPH& vec, size_t k) {

        using Element = std::pair<uint32_t, float>;

        utils::SortedMaxPairSizeDequeR<uint32_t, float> maxDists(k);
        for (SparseVecSPH::InnerIterator it(vec); it; ++it) {
            const float value = it.value();
            const uint32_t index = static_cast<uint32_t>(it.index());
            maxDists.insert(index, value);
        }

        maxDists.prune();

        // Convert priority queue to vector in ascending order
        std::vector<Element> result;
        result.reserve(k);

        for (const auto& pairs : maxDists.getData())
            result.push_back(pairs);

        // We'd like the elements to be sorted by index
        std::sort(result.begin(), result.end(),
            [](const Element& a, const Element& b) {
                return a.first < b.first;
            });

        return result;
    }

    std::vector<std::pair<uint32_t, float>> findBottomK(const SparseVecSPH& vec, size_t k) {

        using Element = std::pair<uint32_t, float>;

        utils::SortedMinPairSizeDequeR<uint32_t, float> minDists(k);
        for (SparseVecSPH::InnerIterator it(vec); it; ++it) {
            const float value = it.value();
            const uint32_t index = static_cast<uint32_t>(it.index());
            minDists.insert(index, value);
        }

        minDists.prune();

        // Convert priority queue to vector in ascending order
        std::vector<Element> result;
        result.reserve(k);

        for (const auto& pairs : minDists.getData())
            result.push_back(pairs);

        // We'd like the elements to be sorted by index
        std::sort(result.begin(), result.end(),
            [](const Element& a, const Element& b) {
                return a.first < b.first;
            });

        return result;
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> createSparseMatrixFromVectors(const SparseMatSPH& sparseVectors, bool verbose)
    {
        if(verbose)
            Log::info("createSparseMatrixFromVectors");

        const size_t rows = sparseVectors.size();

        // Calculate average non-zeros per row
        size_t nonZeros = 0;
        for (size_t row = 0; row < rows; ++row)
            nonZeros += sparseVectors[row].nonZeros();

        // Create a vector of triplets
        std::vector<Eigen::Triplet<float, SparseVecSPH::Index>> triplets;
        triplets.reserve(nonZeros);

        for (size_t row = 0; row < rows; ++row) {
            for (SparseVecSPH::InnerIterator it(sparseVectors[row]); it; ++it) {
                triplets.emplace_back(static_cast<SparseVecSPH::Index>(row), it.index(), it.value());
            }
        }

        // Create the sparse matrix from the combined triplets
        Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatrix(rows, rows);
        sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
        sparseMatrix.makeCompressed();

        return sparseMatrix;
    }

    // Convert sparse matrix rows to vector of sparse vectors
    SparseMatSPH matrixToSparseVectors(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& matrix, bool verbose) {
        if(verbose)
            Log::info("matrixToSparseVectors");

        Eigen::Index numRows = matrix.rows();
        SparseMatSPH result;
        result.resize(numRows);

        // Ensure matrix is in compressed format for efficient row iteration
        if (!matrix.isCompressed()) {
            matrix.makeCompressed();
        }

        utils::ProgressBar progress(numRows, verbose);
        SPH_PARALLEL
        for (Eigen::Index i = 0; i < numRows; ++i) {
            SparseVecSPH& rowVector = result[i];
            rowVector.resize(numRows);

            // Count non-zeros in this row for efficient memory allocation
            size_t nonZeros = 0;
            for (Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator it(matrix, i); it; ++it)
                nonZeros++;

            // Reserve space for the non-zeros
            rowVector.reserve(nonZeros);

            // Copy elements from the matrix row to the sparse vector
            for (Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator it(matrix, i); it; ++it)
                rowVector.insert(it.col()) = it.value();

            SPH_PARALLEL_CRITICAL
            progress.update();
        }

        progress.finish();

        return result;
    }

    template<typename T>
    static std::vector<std::pair<T, T>> divideVectorIntoBlocks(T x, T n) {
        std::vector<std::pair<T, T>> blocks;
        if (n <= 0 || x <= 0) {
            return blocks;
        }

        T baseBlockSize = x / n;
        T remainder = x % n; // Remaining elements to distribute among matrixBlocks

        T start = 0;
        for (T i = 0; i < n; ++i) {
            T blockSize = baseBlockSize + (i < remainder ? 1 : 0); // Add 1 to the first 'remainder' matrixBlocks
            T end = start + blockSize;

            blocks.emplace_back(start, end);

            start = end; // Next block starts after the current block
        }

        return blocks;
    }

    template<typename T>
    static std::vector<std::pair<T, T>> createComputeBlockCombinations(T n) {
        std::vector<std::pair<T, T>> combinations;
        for (T i = 0; i < n; ++i) {
            for (T j = i; j < n; ++j) {
                combinations.emplace_back(i, j);
            }
        }
        return combinations;
    }

    template<typename T>
    static ankerl::unordered_dense::map<std::pair<T, T>, T, ankerl::unordered_dense::hash<std::pair<T, T>>> createComputeBlockMap(T n) {
        ankerl::unordered_dense::map<std::pair<T, T>, T, ankerl::unordered_dense::hash<std::pair<T, T>>> combinations;
        T counter = static_cast<T>(0);
        for (T i = 0; i < n; ++i) {
            for (T j = i; j < n; ++j) {
                combinations[{ i, j }] = counter;
                counter++;
            }
        }
        return combinations;
    }

    template<typename T>
    static std::vector<std::pair<T, T>> createBlockCombinationsAssign(T n) {
        std::vector<std::pair<T, T>> combinations;
        for (T i = 0; i < n; ++i) {
            for (T j = 0; j < n; ++j) {
                if (j < i)
                    combinations.emplace_back(j, i);
                else
                    combinations.emplace_back(i, j);
            }
        }
        return combinations;
    }

    template<typename T>
    static std::vector<std::pair<T, T>> createBlockOffsetsAssign(T n, const std::vector<std::pair<T, T>>& blocks) {
        std::vector<std::pair<T, T>> combinations;
        for (T i = 0; i < n; ++i) {
            for (T j = 0; j < n; ++j) {
                combinations.emplace_back(blocks[i].first, blocks[j].first);
            }
        }

        return combinations;
    }

    template<typename T>
    static std::vector<bool> createTransposeAssign(T n) {
        std::vector<bool> combinations;
        for (T i = 0; i < n; ++i) {
            for (T j = 0; j < n; ++j) {
                if (j < i)
                    combinations.emplace_back(true);
                else
                    combinations.emplace_back(false);
            }
        }

        return combinations;
    }

    // see https://godbolt.org/z/oTPzKz5rK
    template<typename T>
    static std::vector<std::vector<std::pair<T, T>>> createComputeCombinations(T n) {
        std::vector<std::vector<std::pair<T, T>>> combinations;
        for (T i = 0; i < n; ++i) {
            auto& cmbns = combinations.emplace_back();
            for (T j = i; j < n; ++j) {
                cmbns.push_back({ i, j });
            }
        }
        return combinations;
    }

    template<typename T>
    static std::vector<std::vector<std::pair<T, T>>> createOffsets(T n, const std::vector<std::pair<T, T>>& blocks) {
        std::vector<std::vector<std::pair<T, T>>> combinations;
        for (T i = 0; i < n; ++i) {
            auto& cmbns = combinations.emplace_back();
            for (T j = 0; j < n; ++j) {
                cmbns.emplace_back(blocks[i].first, blocks[j].first);
            }
        }

        return combinations;
    }

    SparseMatHDI createSimilarities(const SparseMatSPH& input, int64_t k, float pruneVal, int verbose, const std::optional<std::reference_wrapper<const vvui64>> componentToPixelMap)
    {
        size_t numPoints = input.size();
        SparseMatHDI output(numPoints);

        if (numPoints > 5'000)
        {
            const Eigen::Index blockSize = 1'000;

            // probdist is NOT symmetric
            output = createSimilaritiesHDI(input, k, pruneVal, verbose, blockSize, componentToPixelMap); // 500
        }
        else
        {
            // similarity matrix is symmetric
            const SparseMatSPH randomWalksDistances = createSimilaritiesEigen(input, pruneVal, verbose, 0, componentToPixelMap);

            utils::printSparseMatrixStats(utils::sparseMatrixStats(randomWalksDistances), "randomWalksDistances");

            // probdist is NOT symmetric
            Log::info("LevelSimilarities::computeProbDist: convert to HDILib data structure");
            SPH_PARALLEL
                for (int64_t i = 0; i < static_cast<int64_t>(numPoints); ++i)
                    utils::convertEigenSparseVecToHDILibSparseVec(randomWalksDistances[i], k, output[i], /*top=*/ false); // 500

            if (verbose > 4 && numPoints < 500)
            {
                utils::printSparseMatrixAsDense(randomWalksDistances);
                utils::printSparseMatrixAsDense(output);
            }
        }
        return output;
    }


    SparseMatSPH createSimilaritiesEigen(const SparseMatSPH& input, float pruneVal, int verbose, Eigen::Index blockSize, const std::optional<std::reference_wrapper<const vvui64>> componentToPixelMap)
    {
        using SparseMatrix = Eigen::SparseMatrix<float, Eigen::RowMajor, int>;

        if (verbose > 2)
        {
            const auto inStats = sparseMatrixStats(input);
            printSparseMatrixStats(inStats, "Matrix similarities input");
        }

        SparseMatrix intermediate = utils::createSparseMatrixFromVectors(input, verbose > 1).pruned(pruneVal);
        intermediate.data().squeeze();

        Eigen::Index numRows = intermediate.rows();
        Eigen::Index numCols = intermediate.cols();

        assert(static_cast<size_t>(numRows) == input.size());
        assert(numRows == numCols);

        if (verbose > 1)
            Log::info("createSimilaritiesEigen: create sqrt sparse mat");

        // Compute sqrt of each element
        intermediate = intermediate.cwiseSqrt();

        if (componentToPixelMap.has_value())
        {
            if (verbose > 1)
                Log::info("createSimilaritiesEigen: weight with component size");

            const vvui64& componentToPixel = componentToPixelMap.value();
            assert(componentToPixel.size() == static_cast<size_t>(numRows));

            for (Eigen::Index row = 0; row < numRows; row++)
                for (SparseMatrix::InnerIterator it(intermediate, row); it; ++it)
                    it.valueRef() *= std::sqrt(static_cast<float>(componentToPixel[row].size()));

        }

        SparseMatSPH output;

        bool parallel = blockSize > 0;

        if (parallel && numRows < blockSize)
        {
            Log::warn("createSimilaritiesEigen: asked for parallel but blockSize {0} is smaller than numRows {1}. Defaulting to single-thread.", blockSize, numRows);
            parallel = false;
        }

        if(parallel)
        {
            Eigen::Index numBlocks = numRows / blockSize;

            std::vector<Eigen::Block<SparseMatrix>> block_views;

            const auto matrixBlocks = divideVectorIntoBlocks(numRows, numBlocks);
            const auto blockCombinationsCompute = createComputeBlockCombinations(numBlocks);
            const auto blockMap = createComputeBlockMap(numBlocks);
            const auto blockCombinationsAssign = createBlockCombinationsAssign(numBlocks);
            const auto blockOffsets = createBlockOffsetsAssign(numBlocks, matrixBlocks);
            const auto transposeAssign = createTransposeAssign(numBlocks);

            assert(static_cast<size_t>(numBlocks) == matrixBlocks.size());

            for (const auto& block: matrixBlocks)
            {
                Eigen::Index start_row = block.first;
                Eigen::Index end_row = block.second;
                block_views.emplace_back(intermediate.block(start_row, 0, end_row - start_row, numCols));
            }

            output.resize(numRows);

            SPH_PARALLEL
            for (Eigen::Index i = 0; i < numRows; ++i)
                output[i].resize(numRows);

            std::vector<SparseMatrix> blockResults(blockCombinationsCompute.size());

            ProgressBar progress(blockCombinationsCompute.size(), verbose > 0);

            SPH_PARALLEL
            for (size_t i = 0; i < blockCombinationsCompute.size(); i++)
            {
                const Eigen::Index blockRow = blockCombinationsCompute[i].first;
                const Eigen::Index blockCol = blockCombinationsCompute[i].second;

                blockResults[i] = (block_views[blockRow] * block_views[blockCol].transpose()).pruned(pruneVal);

                if (!blockResults[i].isCompressed()) {
                    blockResults[i].makeCompressed();
                }

                SPH_PARALLEL_CRITICAL
                progress.update();
            }

            progress.finish();
            progress.reset(blockCombinationsAssign.size());

            for (size_t blockID = 0; blockID < blockCombinationsAssign.size(); blockID++)
            {
                auto& blockCombination = blockCombinationsAssign[blockID];

                if (!blockMap.contains(blockCombination))
                    Log::error("?");

                auto& blockKey = blockMap.at(blockCombination);
                SparseMatrix currentBlock = transposeAssign[blockID] ? blockResults[blockKey].transpose() : blockResults[blockKey];

                const Eigen::Index offsetRow = blockOffsets[blockID].first;
                const Eigen::Index offsetCol = blockOffsets[blockID].second;

                Eigen::Index numRowsBlock = currentBlock.rows();

                SPH_PARALLEL
                for (Eigen::Index blockRow = 0; blockRow < numRowsBlock; ++blockRow) {
                    auto& rowVector = output[offsetRow + blockRow];

                    // Count non-zeros in this row for efficient memory allocation
                    size_t nonZeros = 0;
                    for (SparseMatrix::InnerIterator it(currentBlock, blockRow); it; ++it) {
                        nonZeros++;
                    }

                    // Reserve space for the non-zeros
                    rowVector.reserve(rowVector.nonZeros() + nonZeros);

                    // Copy elements from the matrix row to the sparse vector
                    for (SparseMatrix::InnerIterator it(currentBlock, blockRow); it; ++it) {
                        rowVector.insert(offsetCol + it.col()) = it.value();
                    }
                }
                progress.update();
            }

            progress.finish();
        }
        else
        {
            // multiply with transpose self
            // A * A^T accesses A by rows first
            // A^T * A accesses A by columns first
            SparseMatrix multiplied(numRows, numRows);
            multiplied.reserve(Eigen::VectorXf::Constant(numRows, (static_cast<float>(intermediate.nonZeros()) / numRows) * 2.f)); // Estimate density

            multiplied = (intermediate * intermediate.transpose()).pruned(pruneVal);
            output = utils::matrixToSparseVectors(multiplied, false);

            utils::printSparseMatrixStats(utils::sparseMatrixStats(output), "randomWalksDistances");
        }

        SPH_PARALLEL
        for (size_t i = 0; i < output.size(); ++i) {
            for (SparseVecSPH::InnerIterator it(output[i]); it; ++it) {

                // ignore diagonal (which is == 1)
                if (i == static_cast<size_t>(it.index()))
                    continue;

                it.valueRef() = -1.f * std::log(it.value());
            }
        }

        removeDiagonalElements(output);

        if (verbose > 2)
        {
            const auto outStats = sparseMatrixStats(output);
            printSparseMatrixStats(outStats, "Matrix similarities output");
        }

        return output;
    }

    SparseMatHDI createSimilaritiesHDI(const SparseMatSPH& input, int64_t k, float pruneVal, int verbose, Eigen::Index blockSize, const std::optional<std::reference_wrapper<const vvui64>> componentToPixelMap)
    {
        using SparseMatrix = Eigen::SparseMatrix<float, Eigen::RowMajor, int>;
        using SortedQueue = utils::SortedMinPairSizeDequeR<uint32_t, float>;
        using Element = std::pair<uint32_t, float>;

        if (verbose > 2)
        {
            const auto inStats = sparseMatrixStats(input);
            printSparseMatrixStats(inStats, "Matrix similarities input");
        }

        SparseMatrix intermediate = utils::createSparseMatrixFromVectors(input, verbose > 1).pruned(pruneVal);
        intermediate.data().squeeze();

        Eigen::Index numRows = intermediate.rows();
        Eigen::Index numCols = intermediate.cols();

        assert(static_cast<size_t>(numRows) == input.size());
        assert(numRows == numCols);

        if (verbose > 1)
            Log::info("createSimilaritiesHDI: create sqrt sparse mat and blocks");

        // Compute sqrt of each element
        intermediate = intermediate.cwiseSqrt();

        if (componentToPixelMap.has_value())
        {
            if (verbose > 1)
                Log::info("createSimilaritiesHDI: weight with component size");

            const vvui64& componentToPixel = componentToPixelMap.value();
            assert(componentToPixel.size() == static_cast<size_t>(numRows));

            for (Eigen::Index row = 0; row < numRows; row++)
                for (SparseMatrix::InnerIterator it(intermediate, row); it; ++it)
                    it.valueRef() *= static_cast<float>(std::sqrt(componentToPixel[row].size()));

        }

        Eigen::Index numBlocks = 0;

        if (blockSize != 0)
            numBlocks = numRows / blockSize;
        else
            numBlocks = numRows / std::min(numRows, static_cast<Eigen::Index>(1000));

        numBlocks = std::max(numBlocks, static_cast<Eigen::Index>(1));

        assert(numBlocks > 0);

        const auto matrixBlocks = divideVectorIntoBlocks(numRows, numBlocks);
        const auto blockRows = createComputeCombinations(numBlocks);
        const auto offsets = createOffsets(numBlocks, matrixBlocks);
        const size_t numItems = numBlocks * (numBlocks + 1) / 2;

        assert(static_cast<size_t>(numBlocks) == matrixBlocks.size());

        // Group sparse matrix rows into block of size (end_row - start_row) * numCols
        std::vector<Eigen::Block<SparseMatrix>> block_views;
        for (const auto& block : matrixBlocks)
        {
            Eigen::Index start_row = block.first;
            Eigen::Index end_row = block.second;
            block_views.emplace_back(intermediate.block(start_row, 0, end_row - start_row, numCols));
        }

        std::vector<SortedQueue> maxValQueues(numRows, k);

        auto addValueToQueue = [&maxValQueues](const SparseMatrix& currentBlock, const Eigen::Index& blockRow, const Eigen::Index& offsetRow, const Eigen::Index& offsetCol) {
            const Eigen::Index globalRow = offsetRow + blockRow;
            auto& maxValQueue = maxValQueues[globalRow];

            for (SparseMatrix::InnerIterator it(currentBlock, blockRow); it; ++it) {
                const Eigen::Index globalCol = offsetCol + it.col();

                // ignore diagonal (which is == 1)
                if (globalCol == globalRow)
                    continue;

                maxValQueue.insert(static_cast<uint32_t>(globalCol), it.value());
            }

            };

        if (verbose > 1)
            Log::info("createSimilaritiesHDI: multiply sparse matrix blocks and save top k values");

        ProgressBar progress(numItems);

        for (size_t i = 0; i < blockRows.size(); i++)
        {
            auto& blockCombinations = blockRows[i];
            const size_t numCombinations = blockCombinations.size();
            std::vector<SparseMatrix> blockResults(numCombinations);

            SPH_PARALLEL
            for (size_t blockID = 0; blockID < numCombinations; blockID++)
            {
                const Eigen::Index blockRow = blockCombinations[blockID].first;
                const Eigen::Index blockCol = blockCombinations[blockID].second;
                auto& blockResult = blockResults[blockID];

                blockResult = (block_views[blockRow] * block_views[blockCol].transpose()).pruned(pruneVal);
                blockResult.data().squeeze();

                if (!blockResult.isCompressed())
                    blockResult.makeCompressed();
            }

            SPH_PARALLEL
            for (size_t blockID = 0; blockID < numCombinations; blockID++)
            {
                auto& currentBlock = blockResults[blockID];
                float* values = currentBlock.valuePtr();
                for (Eigen::Index k = 0; k < currentBlock.nonZeros(); ++k) {
                    if (values[k] == 1.f)
                        continue;
                    values[k] = -1.f * std::log(values[k]);
                }
            }

            for (size_t blockID = 0; blockID < numCombinations; blockID++)
            {
                auto& currentBlock = blockResults[blockID];

                Eigen::Index blockRow = blockCombinations[blockID].first;
                Eigen::Index blockCol = blockCombinations[blockID].second;

                Eigen::Index offsetRow = offsets[blockRow][blockCol].first;
                Eigen::Index offsetCol = offsets[blockRow][blockCol].second;

                SPH_PARALLEL
                for (Eigen::Index blockRow = 0; blockRow < currentBlock.rows(); ++blockRow) {
                    addValueToQueue(currentBlock, blockRow, offsetRow, offsetCol);
                }

                if (blockID == 0)
                    continue;

                std::swap(blockRow, blockCol);

                SparseMatrix currentBlockTranpose = currentBlock.transpose();

                offsetRow = offsets[blockRow][blockCol].first;
                offsetCol = offsets[blockRow][blockCol].second;

                SPH_PARALLEL
                for (Eigen::Index blockRow = 0; blockRow < currentBlockTranpose.rows(); ++blockRow) {
                    addValueToQueue(currentBlockTranpose, blockRow, offsetRow, offsetCol);
                }

            }

            blockResults.clear();
            progress.updateBy(numCombinations);
        }
        progress.finish();

        intermediate.setZero();
        intermediate.data().squeeze();

        if (verbose > 1)
            Log::info("createSimilaritiesHDI: convert in HDILib mat");

        SparseMatHDI output(numRows);

        progress.reset(numRows);

        SPH_PARALLEL
        for (int64_t row = 0; row < static_cast<int64_t>(numRows); ++row)
        {
            auto& maxValQueue = maxValQueues[row];
            maxValQueue.prune();

            // Convert priority queue to vector in ascending order
            std::vector<Element>& hdiVec = output[row].memory();
            hdiVec.reserve(k);

            for (const auto& pairs : maxValQueue.getData())
                hdiVec.push_back(pairs);

            // We'd like the elements to be sorted by index
            std::sort(hdiVec.begin(), hdiVec.end(),
                [](const Element& a, const Element& b) {
                    return a.first < b.first;
                });

            // normalize such that each row sums to one
            double sum = 0;
            for (const auto& elem : hdiVec) {
                sum += elem.second;
            }
            for (auto& elem : hdiVec) {
                elem.second /= static_cast<float>(sum);
            }

            SPH_PARALLEL_CRITICAL
                progress.update();
        }

        progress.finish();

        return output;
    }

    SparseMatSPH createSimilaritiesSPH(const SparseMatSPH& input, float pruneVal, int verbose)
    {
        if (verbose > 2)
        {
            const auto inStats = sparseMatrixStats(input);
            printSparseMatrixStats(inStats, "Matrix similarities input");
        }

        SparseMatSPH output;
        output.resize(input.size());

        // Equivalent but slower
        // auto dist = [&input](size_t id1, size_t id2) -> float {
        //     // assumes probDist1 and probDist2 are normalized
        //     const auto& probDist1 = input[id1];
        //     const auto& probDist2 = input[id2];

        //     float bc = 0.f;

        //     for (Eigen::SparseVector<float>::InnerIterator it1(probDist1); it1; ++it1) {
        //         const float value1 = it1.value();
        //         assert(value1 > 0.0f);
        //         const float value2 = probDist2.coeff(it1.index()); // get the corresponding value from normDist2, 0 if not present
        //         if (value2 > 0.0f) {
        //             bc += std::sqrt(value1 * value2);
        //         }
        //     }
        //     return bc;
        //     };

        SparseMatSPH sqrt_vec;
        std::copy(input.begin(), input.end(), std::back_inserter(sqrt_vec));

        SPH_PARALLEL
        for (size_t i = 0; i < input.size(); ++i) {
            float* values = sqrt_vec[i].valuePtr();
            Eigen::Index nonZeros = sqrt_vec[i].nonZeros();
            for (Eigen::Index j = 0; j < nonZeros; ++j) {
                values[j] = std::sqrt(values[j]);
            }
        }

        // Equivalent but slower
        //auto dot = [](const SparseVecSPH& vec1, const SparseVecSPH& vec2) -> float {
        //    float bc = 0.f;

        //    for (SparseVecSPH::InnerIterator it1(vec1); it1; ++it1) {
        //        assert(it1.value() > 0.0f);
        //        const float value2 = vec2.coeff(it1.index()); // get the corresponding value from normDist2, 0 if not present
        //        if (value2 > 0.0f) {
        //            bc += it1.value() * value2;
        //        }
        //    }

        //    return bc;

        //    };

        if (verbose > 1)
            Log::info("createSimilaritiesSPH: sparse vector multiplication");

        utils::ProgressBar progress(sqrt_vec.size(), verbose > 0);

        SPH_PARALLEL
        for (size_t i = 0; i < sqrt_vec.size(); ++i) {
            output[i].resize(input.size());
        }

        SPH_PARALLEL
        for (size_t i = 0; i < sqrt_vec.size(); ++i) {
            for (size_t j = i + 1; j < sqrt_vec.size(); ++j) {
                float dotProduct = sqrt_vec[i].dot(sqrt_vec[j]);
                //float dotProduct = dot(sqrt_vec[i], sqrt_vec[j]);

                if (dotProduct < pruneVal)
                    continue;

                float out_val = -1.f * std::log(dotProduct);

                SPH_PARALLEL_CRITICAL
                {
                    output[i].insert(j) = out_val;
                    output[j].insert(i) = out_val;
                }
            }

            SPH_PARALLEL_CRITICAL
            progress.update();
        }

        // make symmetric from upper triangle
        //for (size_t i = 0; i < output.size(); ++i) {
        //    for (SparseVecSPH::InnerIterator it(output[i]); it; ++it) {
        //        if (it.index() < i)
        //            continue;
        //        output[it.index()].insert(i) = it.value();
        //    }
        //}

        progress.finish();

        if (verbose > 2)
        {
            const auto outStats = sparseMatrixStats(output);
            printSparseMatrixStats(outStats, "Matrix similarities output");
        }

        return output;
    }

    void convertEigenSparseVecToHDILibSparseVec(const SparseVecSPH& eigenVec, int64_t k, SparseVecHDI& hdiVec, bool top)
    {
        if (eigenVec.size() == 0)
        {
            Log::warn("convertEigenSparseVecToHDILibSparseVec: input eigenVec is empty");
            return;
        }

        // take k largest similarities
        if(top)
            hdiVec.memory() = findTopK(eigenVec, k);
        else
            hdiVec.memory() = findBottomK(eigenVec, k);

        assert(hdiVec.memory().size() >= static_cast<size_t>(k) ? hdiVec.memory().size() == static_cast<size_t>(k) : true);

        // normalize such that each row sums to one
        double sum = 0;
        for (const auto& elem : hdiVec) {
            sum += elem.second;
        }

        if (sum <= 0)
        {
            Log::warn("convertEigenSparseVecToHDILibSparseVec: output hdiVec is all zero");
            return;
        }

        for (auto& elem : hdiVec) {
            elem.second /= static_cast<float>(sum);
        }

#if SPH_DEBUG
        float sumTest = std::accumulate(hdiVec.begin(), hdiVec.end(), 0.f, [](float s, const std::pair<uint32_t, float>& pp) { return std::move(s) + pp.second; });
        assert(std::abs(sumTest - 1.f) < 0.001f);
#endif
    }

    void convertSparseMatSPHToSparseMatHDI(const SparseMatSPH& matSPH, SparseMatHDI& matHDI)
    {
        matHDI.resize(matSPH.size());

        SPH_PARALLEL
        for (size_t i = 0; i < matSPH.size(); ++i)
        {
            const auto& rowFrom = matSPH[i];
            auto& rowTo = matHDI[i].memory();

            for (SparseVecSPH::InnerIterator it(rowFrom); it; ++it)
                rowTo.push_back({ static_cast<uint32_t>(it.index()) , it.value() });
        }
    }

    bool isSymmetric(const SparseMatHDI& matSPH)
    {
        uint32_t numPoints = static_cast<uint32_t>(matSPH.size());
        for (uint32_t j = 0; j < numPoints; ++j) {
            for (auto& e : matSPH[j]) {
                const auto i = e.first;
                const auto v1 = e.second;

                const auto v2 = matSPH[i].find(j);
                
                if (v2 == matSPH[i].cend())
                    return false;

                if (v1 != v2->second)
                    return false;
            }
        }

        return true;
    }
        
    bool isSame(const SparseMatHDI& a, const SparseMatHDI& b)
    {
        // same number rows
        if (a.size() != b.size())
            return false;

        uint32_t numPoints = static_cast<uint32_t>(a.size());
        for (uint32_t j = 0; j < numPoints; ++j) {

            const auto& rowA = a[j];
            const auto& rowB = b[j];

            // same number columns
            if (rowA.size() != rowB.size())
                return false;

            for (auto& e : rowA) {
                const auto i = e.first;
                const auto v1 = e.second;

                // same entries
                const auto v2 = rowB.find(i);
                if (v2 == rowB.cend())
                    return false;

                if (v1 != v2->second)
                    return false;
            }
        }

        return true;

    }

} // namespace sph::utils

