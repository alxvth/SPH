#include "GraphNormalization.hpp"

#include "CommonDefinitions.hpp"
#include "Graph.hpp"
#include "Logger.hpp"
#include "Math.hpp"
#include "PrintHelper.hpp"
#include "ProgressBar.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

#if SPH_RELEASE
#include <omp.h>
#endif

#include <hdi/data/map_mem_eff.h>
#include <hdi/utils/math_utils.h>

#include <Eigen/SparseCore>     // Eigen::SparseVector

#pragma warning(disable:4244) // umappp internal: conversion from 'type1' to 'type2'
#pragma warning(disable:4267) // umappp internal: conversion from 'size_t' to 'type'
#pragma warning(disable:4305) // umappp internal: 'initializing': truncation from 'double' to 'Float_' (options are all double internally)
#include <umappp/NeighborList.hpp>
#include <umappp/neighbor_similarities.hpp>
#pragma warning(default:4305)
#pragma warning(default:4267)
#pragma warning(default:4244)

namespace sph::utils {

    template <typename T>
    void computeGaussianDistributions(std::vector<T>& similarities, float fixedPerplexity, int ignore)
    {
        const auto np = similarities.size();

        auto getCurrentValues = [&similarities](int64_t p) -> std::vector<float> {
            std::vector<float> vals;
            
            if constexpr (std::is_same_v<T, SparseVecSPH>)
            {
                vals.reserve(similarities[p].nonZeros());
                for (SparseVecSPH::InnerIterator it(similarities[p]); it; ++it)
                    vals.push_back(it.value());
            }
            else
            {
                vals.reserve(similarities[p].size());
                for (auto it = similarities[p].begin(); it != similarities[p].end(); ++it)
                    vals.push_back(it->second);
            }

            return vals;
            };

        uint64_t noSigmaCounter = 0;
        utils::ProgressBar progress(np);

        SPH_PARALLEL
        for (int64_t p = 0; p < static_cast<int64_t>(np); ++p) {

            std::vector<float> input_vector = getCurrentValues(p);

            const int64_t nn = input_vector.size();

            if (nn <= 1)
                continue;

            float perplexity = 0;
            if (fixedPerplexity > 0)
                perplexity = fixedPerplexity;
            else
                perplexity = nn / 3.f;

            assert(perplexity > 0);

            std::vector<float> condProb(nn, -1.f);

            double sigma = hdi::utils::computeGaussianDistributionWithFixedPerplexity<std::vector<float>>(
                input_vector.begin(),
                input_vector.end(),
                condProb.begin(),
                condProb.end(),
                perplexity,
                /* max iter */      200,
                /* tolerance */     1e-6,
                /* ignore index */  ignore
            );

            constexpr double minSigma = 0.001;

            if (sigma < minSigma)
            {
                std::copy(input_vector.begin(), input_vector.end(), condProb.begin());

                auto allZero = [](std::vector<float>& probs) -> bool {
                    return std::all_of(probs.cbegin(), probs.cend(), [](const auto val) { return val == 0; });
                    };

                auto assignSameProb = [&nn](std::vector<float>& probs) -> void {
                    const float sameProb = 1.f / (nn - 1);
                    std::fill(probs.begin(), probs.end(), sameProb);
                    };

                if (allZero(condProb))
                    assignSameProb(condProb);
                else
                {
                    // First norm to sum() == 1, invert, then ensure sum() == 1
                    utils::normalizeUnit(condProb.begin(), condProb.end());
                    std::transform(condProb.begin(), condProb.end(), condProb.begin(), [](float element) {
                        return 1.0f - element;
                        });

                    if (allZero(condProb))
                        assignSameProb(condProb);
                    else
                    {
                        if (ignore >= 0)
                            condProb[0] = 0.f;
                        utils::normalizeUnit(condProb.begin(), condProb.end());
                    }
                }
            }

            int64_t k = 0;
            constexpr float minVal = 1.0e-10f;
            if constexpr (std::is_same_v<T, SparseVecSPH>)
            {
                for (SparseVecSPH::InnerIterator it(similarities[p]); it; ++it)
                {
                    if (condProb[k] < minVal)
                        continue;
                    it.valueRef() = condProb[k];
                    k++;
                }
            }
            else
            {
                for (auto it = similarities[p].begin(); it != similarities[p].end(); ++it)
                {
                    if (condProb[k] < minVal)
                        continue;
                    it->second = condProb[k];
                    k++;
                }
            }

#if SPH_DEBUG
            {
                constexpr double eps = 0.01f;

                double sum = 0;
                for (const auto val : condProb)
                    sum += val;

                if constexpr (std::is_same_v<T, SparseVecSPH>)
                {
                    if (std::isnan(sum) || std::abs(sum - 1.0) >= eps)
                    {
                        Log::warn("computeGaussianDistribution:: {0}", p);
                        utils::printSparseVector(similarities[p], true);
                        utils::print(std::span{ input_vector.begin(), input_vector.end() });
                    }
                }

                assert(std::abs(sum - 1.0) < eps);

                SPH_PARALLEL_CRITICAL
                if (sigma < minSigma)
                    noSigmaCounter++;
            }
#endif // SPH_DEBUG

            SPH_PARALLEL_CRITICAL
            progress.update();
        }
        progress.finish();

        Log::debug("computeGaussianDistribution:: high-dimensional conditional probability distribution set to uniform for {0} points ({1:.4g} %)", noSigmaCounter, 100.f * noSigmaCounter / np);
    }
    template void computeGaussianDistributions<hdi::data::MapMemEff<uint32_t, float>>(std::vector<hdi::data::MapMemEff<uint32_t, float>>&, float, int);
    template void computeGaussianDistributions<SparseVecSPH>(std::vector<SparseVecSPH>&, float, int);

    template <typename T>
    void computeGaussianDistributions(const GraphBaseInterface& graphView, std::vector<T>& similarities, float fixedPerplexity)
    {
        const auto np = graphView.getNumPoints();
        similarities.resize(np);

        if constexpr (std::is_same_v<T, SparseVecSPH>)
        {
            for (int64_t p = 0; p < static_cast<int64_t>(np); ++p)
            {
                auto& sims = similarities[p];
                sims.resize(np);
                sims.reserve(graphView.getK(p) - 1);
            }
        }

        uint64_t noSigmaCounter = 0;
        utils::ProgressBar progress(np);

        // as in HierarchicalSNE::computeFMC
        SPH_PARALLEL
        for (int64_t p = 0; p < static_cast<int64_t>(np); ++p) {
            //It could be that the point itself is not the nearest one if two points are identical... I want the point itself to be the first one!
            if (graphView.getNeighborN(p, 0) != p) {
                Log::warn("computeGaussianDistribution:: The first nn of {0} should be the point itself but is instead: {1}", p, graphView.getNeighborN(p, 0));
            }

            // Conditional probability distribution in the high-dimensional space
            // first neighbor is point itself, ignore it
            const int64_t nn = graphView.getK(p);
            const int64_t nn_eff = nn - 1;
            std::vector<float> condProb(nn, -1.f);

            float perplexity = 0;
            if (fixedPerplexity > 0)
                perplexity = fixedPerplexity;
            else
                perplexity = nn_eff / 3.f;

            assert(graphView.getDistances(p).size() == static_cast<size_t>(nn));
            assert(graphView.getDistances(p)[0] == 0.f);

            const auto knnDistancesPoint    = graphView.getDistances(p);
            const auto knnDistancesOffset   = knnDistancesPoint.data() - graphView.getKnnDistances().data();
            const auto knnDistancesBegin    = graphView.getKnnDistances().cbegin() + knnDistancesOffset;
            const auto knnDistancesEnd      = knnDistancesBegin + nn;

            double sigma = hdi::utils::computeGaussianDistributionWithFixedPerplexity<std::vector<float>>(
                knnDistancesBegin,
                knnDistancesEnd,
                condProb.begin(),
                condProb.end(),
                perplexity,
                /* max iter */      200,
                /* tolerance */     1e-6,
                /* ignore index */  0
            );

            constexpr double minSigma = 0.001;

            if (sigma < minSigma)
            {
                std::copy(knnDistancesBegin, knnDistancesEnd, condProb.begin());

                auto allZero = [](std::vector<float>& probs) -> bool {
                    return std::all_of(probs.cbegin(), probs.cend(), [](const auto val) { return val == 0; });
                };

                auto assignSameProb = [&nn_eff](std::vector<float>& probs) -> void {
                    const float sameProb = 1.f / nn_eff;
                    std::fill(probs.begin(), probs.end(), sameProb);
                };

                if (allZero(condProb))
                    assignSameProb(condProb);
                else
                {
                    // First norm to sum() == 1, invert, then ensure sum() == 1
                    utils::normalizeUnit(condProb.begin(), condProb.end());
                    std::transform(condProb.begin(), condProb.end(), condProb.begin(), [](float element) {
                        return 1.0f - element;
                        });

                    if (allZero(condProb))
                        assignSameProb(condProb);
                    else
                    {
                        condProb[0] = 0.f;
                        utils::normalizeUnit(condProb.begin(), condProb.end());
                    }
                }
            }

            auto& sims = similarities[p];
            auto neighborIDs = graphView.getNeighbors(p);

            for (int64_t n = 1; n < nn; ++n)
            {
                if constexpr (std::is_same_v<T, SparseVecSPH>)
                    sims.coeffRef(static_cast<uint32_t>(neighborIDs[n])) = condProb[n];
                else
                    sims[static_cast<uint32_t>(neighborIDs[n])] = condProb[n];
            }

#if SPH_DEBUG
            {
                double sum = 0;

                for (const auto val : condProb | std::views::drop(1))
                    sum += val;

                constexpr double eps = 0.001;

                if constexpr (std::is_same_v<T, SparseVecSPH>)
                {
                    if (std::isnan(sum) || std::abs(sum - 1.0) >= eps)
                    {
                        Log::warn("computeGaussianDistribution:: {0}, sum is {1}", p, sum);
                        utils::printSparseVector(sims, true);
                        utils::print(std::span{ knnDistancesBegin, knnDistancesEnd });
                    }
                }

                assert(std::abs(sum - 1.0) < eps);

                // elements are decreasing or the same since we assume the input graph distance to be increasing
                auto it = std::adjacent_find(condProb.begin() + 1, condProb.end(), [](float a, float b) {
                return a < b;  // Find if there is any adjacent pair where the previous is less than the next
                    });

                assert(it == condProb.end());

                SPH_PARALLEL_CRITICAL
                if (sigma < minSigma)
                    noSigmaCounter++;
            }
#endif // SPH_DEBUG

            SPH_PARALLEL_CRITICAL
            progress.update();
        }
        progress.finish();

        Log::debug("computeGaussianDistribution:: high-dimensional conditional probability distribution set to uniform for {0} points ({1:.4g} %)", noSigmaCounter, 100.f * noSigmaCounter / np);

    }
    template void computeGaussianDistributions<hdi::data::MapMemEff<uint32_t, float>>(const GraphBaseInterface& graphView, std::vector<hdi::data::MapMemEff<uint32_t, float>>&, float);
    template void computeGaussianDistributions<SparseVecSPH>           (const GraphBaseInterface& graphView, std::vector<SparseVecSPH>&,            float);

    template <typename T>
    void computeLinearDistributions(const GraphBaseInterface& graphView, std::vector<T>& similarities)
    {
        auto np = graphView.getNumPoints();

        similarities.resize(np);

        if constexpr (std::is_same_v<T, SparseVecSPH>)
        {
            auto nn = graphView.getK();
            for (auto& sims : similarities)
            {
                sims.resize(np);
                sims.reserve(nn - 1);
            }
        }

        utils::ProgressBar progress(np);

        SPH_PARALLEL
        for (int64_t p = 0; p < np; ++p) {
            //It could be that the point itself is not the nearest one if two points are identical... I want the point itself to be the first one!
            if (graphView.getNeighborN(p, 0) != p) {
                Log::warn("normalizeDistancesToUnit:: The first nn should be the point itself");
            }

            // Conditional probability distribution in the high-dimensional space
            const auto nn = graphView.getK(p);
            std::vector<float> condProb(nn - 1, -1.f);

            // ignore first entry - it's always 0
            const auto knnDistancesPoint = graphView.getDistances(p);
            const auto neighborIDs = graphView.getNeighbors(p);

            std::copy(knnDistancesPoint.begin() + 1, knnDistancesPoint.end(), condProb.begin());

            // inversion/reciprocal, maps [0, inf] to [1, 0]
            for (auto& prob : condProb)
                prob = invlin(prob);

            // normalize such that sum == 1
            utils::normalizeUnit(condProb.begin(), condProb.end());
            
            // convert to data structure
            if constexpr (std::is_same_v<T, SparseVecSPH>)
                for (int64_t n = 1; n < nn; ++n)
                    similarities[p].coeffRef(static_cast<uint32_t>(neighborIDs[n])) = condProb[n-1];
            else
                for (int64_t n = 1; n < nn; ++n)
                    similarities[p][static_cast<uint32_t>(neighborIDs[n])] = condProb[n-1];

            // check high-dimensional conditional probability distribution
#if SPH_DEBUG
            {
                float sum = 0.f;
                if constexpr (std::is_same_v<T, SparseVecSPH>)
                    sum = similarities[p].sum();
                else
                    sum = std::accumulate(similarities[p].begin(), similarities[p].end(), 0.f, [](float s, const std::pair<uint32_t, float>& pp) { return std::move(s) + pp.second; });

                assert(std::abs(sum - 1.f) < 0.001f);
            }
#endif // SPH_DEBUG

            SPH_PARALLEL_CRITICAL
            progress.update();
        }
        progress.finish();

    }
    template void computeLinearDistributions<hdi::data::MapMemEff<uint32_t, float>>(const GraphBaseInterface& graphView, std::vector<hdi::data::MapMemEff<uint32_t, float>>&);
    template void computeLinearDistributions<SparseVecSPH>(const GraphBaseInterface& graphView, std::vector<SparseVecSPH>&);

    template <typename T>
    void computeExponentialDistributions(std::vector<T>& similarities, int ignore)
    {
        const size_t np = similarities.size();

        umappp::NeighborList<int64_t, float> neighbors;
        neighbors.resize(np);

        // Convert to umappp structure
        utils::ProgressBar progress(np);
    SPH_PARALLEL
        for (size_t p = 0; p < np; p++)
        {
            auto& currentNeighbors = neighbors[p];
            const auto& currentSimilarities = similarities[p];

            if constexpr (std::is_same_v<T, SparseVecSPH>)
            {
                currentNeighbors.reserve(currentSimilarities.nonZeros());
                for (SparseVecSPH::InnerIterator it(currentSimilarities); it; ++it)
                {
                    if (it.index() == ignore)
                        continue;
                    currentNeighbors.push_back({ it.index(), it.value() });
                }
            }
            else
            {
                currentNeighbors.reserve(currentSimilarities.size());
                for (const auto& [id, val]: currentSimilarities)
                {
                    if (id == static_cast<uint32_t>(ignore))
                        continue;
                    currentNeighbors.push_back({ static_cast<uint32_t>(id), val });
                }
            }

            SPH_PARALLEL_CRITICAL
            progress.update();
        }
        progress.finish();

        // Normalize data
        umappp::NeighborSimilaritiesOptions<float> nsOptions;
        // nsOptions.local_connectivity = ; // use default: 1.0
        // nsOptions.bandwidth = ;          // use default: 1.0
        nsOptions.num_threads = omp_get_max_threads();

        umappp::neighbor_similarities(neighbors, nsOptions);

        // Convert back to SPH structure
        progress.reset();

        SPH_PARALLEL
        for (size_t p = 0; p < np; ++p) {
            const auto& neighs = neighbors[p];
            auto& sims = similarities[p];

            if constexpr (std::is_same_v<T, SparseVecSPH>)
            {
                for (const auto& [id, val]: neighs)
                    sims.coeffRef(id) = val;
            }
            else
            {
                auto& storage = sims.memory();

                for (size_t i = 0; i < neighs.size(); ++i)
                {
                    assert(storage[i].first == static_cast<uint32_t>(neighs[i].first));
                    storage[i].second = neighs[i].second;
                }
            }

#if SPH_DEBUG
            {
                float sum = 0.f;

                // check high-dimensional conditional probability distribution
                if constexpr (std::is_same_v<T, SparseVecSPH>)
                    sum = sims.sum();
                else
                    sum = std::accumulate(sims.begin(), sims.end(), 0.f, [](float s, const std::pair<uint32_t, float>& pp) { return std::move(s) + pp.second; });

                assert(std::abs(sum - 1.f) < 0.001f);
            }
#endif // SPH_DEBUG
            SPH_PARALLEL_CRITICAL
            progress.update();
        }

    }
    template void computeExponentialDistributions<hdi::data::MapMemEff<uint32_t, float>>(std::vector<hdi::data::MapMemEff<uint32_t, float>>&, int);
    template void computeExponentialDistributions<SparseVecSPH>(std::vector<SparseVecSPH>&, int);

    template <typename T>
    void computeExponentialDistributions(const GraphBaseInterface& graphView, std::vector<T>& similarities)
    {
        const int64_t np = graphView.getNumPoints();

        umappp::NeighborList<int64_t, float> neighbors;
        neighbors.resize(np);

        // Convert to umappp structure
        utils::ProgressBar progress(np);
        SPH_PARALLEL
        for (int64_t i = 0; i < np; i++)
        {
            const int64_t numK = graphView.getK(i);
            for (int64_t k = 1; k < numK; k++)
            {
                const std::pair<int64_t, float> neighborPair = graphView.getNeighborDistanceN(i, k);
                neighbors[i].push_back({ neighborPair.first, neighborPair.second });
            }
            SPH_PARALLEL_CRITICAL
            progress.update();
        }
        progress.finish();

        // Normalize data
        umappp::NeighborSimilaritiesOptions<float> nsOptions;
        // nsOptions.local_connectivity = ; // use default: 1.0
        // nsOptions.bandwidth = ;          // use default: 1.0
        nsOptions.num_threads = omp_get_max_threads();

        umappp::neighbor_similarities(neighbors, nsOptions);

        // Convert back to SPH structure
        similarities.resize(np);

        if constexpr (std::is_same_v<T, SparseVecSPH>)
        {
            auto nn = graphView.getK();
            for (auto& sims : similarities)
            {
                sims.resize(np);
                sims.reserve(nn - 1);
            }

        }

        progress.reset();

        SPH_PARALLEL
        for (int64_t p = 0; p < np; ++p) {
            const auto& neighs = neighbors[p];
            auto& sims = similarities[p];

            if constexpr (std::is_same_v<T, SparseVecSPH>)
            {
                for (const auto& [id, val]: neighs)
                    sims.insert(id) = val;
            }
            else
            {
                auto& storage = sims.memory();

                for (const auto& [id, val] : neighs)
                    storage.emplace_back(static_cast<uint32_t>(id), val);
            }

#if SPH_DEBUG
            {
                float sum = 0.f;

                // check high-dimensional conditional probability distribution
                if constexpr (std::is_same_v<T, SparseVecSPH>)
                    sum = sims.sum();
                else
                    sum = std::accumulate(sims.begin(), sims.end(), 0.f, [](float s, const std::pair<uint32_t, float>& pp) { return std::move(s) + pp.second; });

                assert(std::abs(sum - 1.f) < 0.001f);
            }
#endif // SPH_DEBUG
            SPH_PARALLEL_CRITICAL
            progress.update();
        }

    }
    template void computeExponentialDistributions<hdi::data::MapMemEff<uint32_t, float>>(const GraphBaseInterface& graphView, std::vector<hdi::data::MapMemEff<uint32_t, float>>&);
    template void computeExponentialDistributions<SparseVecSPH>(const GraphBaseInterface& graphView, std::vector<SparseVecSPH>&);

} // namespace sph::utils

