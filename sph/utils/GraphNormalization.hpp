#pragma once

#include "Logger.hpp"
#include "Settings.hpp"

#include <vector>

// Defines all algorithms for SparseVecSPH and SparseVecHDI
// ensure to include Eigen/SparseCore and hdi/data/map_mem_eff.h respectively

namespace sph::utils {

    struct GraphBaseInterface;

    // distance normalization as in t-SNE, in-place
    template <typename T>
    void computeGaussianDistributions(std::vector<T>& similarities, float fixedPerplexity = -1, int ignore = 0);

    // distance normalization as in t-SNE
    template <typename T>
    void computeGaussianDistributions(const GraphBaseInterface& graphView, std::vector<T>& similarities, float fixedPerplexity = -1);

    // converts the distance (in [0, inf)) to similarities (in [1, 0)) by inverting with 1 / ( 1 + d ) and normalizes such that sum of row is 1
    template <typename T>
    void computeLinearDistributions(const GraphBaseInterface& graphView, std::vector<T>& similarities);

    // distance normalization as in UMAP, in-place
    template <typename T>
    void computeExponentialDistributions(std::vector<T>& similarities, int ignore = 0);

    // distance normalization as in UMAP
    template <typename T>
    void computeExponentialDistributions(const GraphBaseInterface& graphView, std::vector<T>& similarities);

    template <typename T>
    void normalizeKnnDistances(const GraphBaseInterface& graphView, const utils::NormalizationScheme normScheme, std::vector<T>& similarities)
    {
        switch (normScheme)
        {
        case utils::NormalizationScheme::TSNE:
            utils::computeGaussianDistributions<T>(graphView, similarities);
            break;
        case utils::NormalizationScheme::LINEAR:
            utils::computeLinearDistributions<T>(graphView, similarities);
            break;
        case utils::NormalizationScheme::UMAP:
            utils::computeExponentialDistributions<T>(graphView, similarities);
            break;
        default:
            Log::error("normalizeKnnDistances: Norm scheme not handled. Defaulting to TSNE");
            utils::computeGaussianDistributions<T>(graphView, similarities);
        }
    }


} // namespace sph::utils
