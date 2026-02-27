#include "Math.hpp"

#include "Algorithms.hpp"
#include "CommonDefinitions.hpp"
#include "Graph.hpp"
#include "PCA.hpp"
#include "PrintHelper.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <execution>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <thread>
#include <vector>

#if SPH_RELEASE
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/stride.hpp>

#pragma warning(disable:4244) // umappp internal: conversion from 'type1' to 'type2'
#pragma warning(disable:4267) // umappp internal: conversion from 'size_t' to 'type'
#pragma warning(disable:4305) // umappp internal: 'initializing': truncation from 'double' to 'Float_' (options are all double internally)
#include <irlba/Options.hpp>
#include <umappp/spectral_init.hpp>
#include <umappp/NeighborList.hpp>
#pragma warning(default:4305)
#pragma warning(default:4267)
#pragma warning(default:4244)


namespace sph::utils {

    int64_t randomNumberUniform(int64_t min, int64_t max)
    {
        static std::random_device rd;
        static std::default_random_engine gen(rd());

        std::uniform_int_distribution<int64_t> distrib(min, max);
        return distrib(gen);
    }

    // https://godbolt.org/z/hbEvK3aab
    float jaccardCoeff(const SparseVecSPH& A, const SparseVecSPH& B) {
        // Get the non-zero indices of A and B
        std::vector<int64_t> indicesA(A.nonZeros());
        std::vector<int64_t> indicesB(B.nonZeros());

        int64_t sizeA = static_cast<int64_t>(A.nonZeros());
        int64_t sizeB = static_cast<int64_t>(B.nonZeros());

        SPH_PARALLEL
        for (int64_t i = 0; i < sizeA; ++i) {
            indicesA[i] = A.innerIndexPtr()[i];
        }

        SPH_PARALLEL
        for (int64_t i = 0; i < sizeB; ++i) {
            indicesB[i] = B.innerIndexPtr()[i];
        }

        // Sort the indices
        std::sort(indicesA.begin(), indicesA.end());
        std::sort(indicesB.begin(), indicesB.end());

        // Compute intersection, union, and product
        float intersection = 0.f;
        float union_set = 0.f;

        int64_t i = 0, j = 0;

        while (i < sizeA && j < sizeB) {
            if (indicesA[i] < indicesB[j]) {
                ++i;
            }
            else if (indicesA[i] > indicesB[j]) {
                ++j;
            }
            else {
                // Found an intersection
                float valueA = A.coeff(indicesA[i]);
                float valueB = B.coeff(indicesB[j]);

                intersection += std::fmin(valueA, valueB);
                union_set += std::fmax(valueA, valueB);

                ++i;
                ++j;
            }
        }

        return (union_set == 0.f) ? 0.f : intersection / union_set;
    }

    float jaccardCoeff(const std::vector<float>& a, const std::vector<float>& b) {
        assert(a.size() == b.size());

        float intersection = 0.f;
        float union_set = 0.f;

        for (std::size_t i = 0; i < a.size(); ++i) {
            intersection += std::fmin(a[i], b[i]);
            union_set += std::fmax(a[i], b[i]);
        }

        return (union_set == 0.f) ? 0.f : intersection / union_set;
    }

    void clampValues(std::vector<float>& data, float newMin, float newMax)
    {
        assert(newMax >= newMin);

        auto clampDists = [newMin, newMax](float& value) -> void {
            value = std::clamp(value, newMin, newMax);
            };

        std::for_each(SPH_PARALLEL_EXECUTION
            data.begin(),
            data.end(),
            clampDists);

    }

    float computeQuantile(const std::vector<float>& data, const float quantile, const std::vector<float>& ignoreVals, const int interpolation) {
        // local copy of data
        std::vector<float> sorted_data = data;

        if (ignoreVals.size() > 0) {
            for (const float ignoreVal : ignoreVals)
                sorted_data.erase(std::remove(sorted_data.begin(), sorted_data.end(), ignoreVal), sorted_data.end());
        }

        std::sort(SPH_PARALLEL_EXECUTION
            sorted_data.begin(),
            sorted_data.end());

        // Convert percentile to index position
        const float rank = quantile * (sorted_data.size() - 1);
        const size_t lower_idx = static_cast<size_t>(std::floor(rank));
        const size_t upper_idx = static_cast<size_t>(std::ceil(rank));

        // Handle edge cases
        if (lower_idx == upper_idx) {
            return sorted_data[lower_idx];
        }

        const float fraction = rank - lower_idx;
        const float lower_value = sorted_data[lower_idx];
        const float upper_value = sorted_data[upper_idx];

        if (interpolation == 1) {
            return lower_value + (upper_value - lower_value) * fraction;
        }

        return 0.5f * (lower_value + upper_value);
    }

    float symmetricHausdorffDistance(const Eigen::MatrixXf& distanceMatrix) {
        return std::max(
            distanceMatrix.rowwise().minCoeff().maxCoeff(),
            distanceMatrix.colwise().minCoeff().maxCoeff()
        );
    }

    void normalizeData(std::vector<float>& data, size_t numDims)
    {
        for (size_t d = 0; d < numDims; d++)
        {
            auto dimRange = data | ranges::views::drop(d) | ranges::views::stride(numDims);
            normalizeMinMax(dimRange.begin(), dimRange.end());
        }
    }

    void normalizeChannelsUniform(std::vector<float>& data, size_t numDims)
    {
        for (size_t d = 0; d < numDims; d++)
        {
            auto dimRange = data | ranges::views::drop(d) | ranges::views::stride(numDims);
            normalizeUniform(dimRange.begin(), dimRange.end());
        }
    }

    void normalizeStandard(std::vector<float>& data, size_t numDims)
    {
        const size_t numPoints = data.size() / numDims;
        std::vector<float> means = computeMean(data, numDims);
        std::vector<float> stds = computeStd(data, means);

        //for (size_t dim = 0; dim < numDims; ++dim)
        //    Log::info("Dimension {0}: Mean = {1}, Std = {2}", dim, means[dim], stds[dim]);

        for (size_t i = 0; i < numPoints; ++i) {
            for (size_t dim = 0; dim < numDims; ++dim) {
                data[numDims * i + dim] = (data[numDims * i + dim] - means[dim]) / stds[dim];
            }
        }
    }

    std::pair<std::vector<float>, bool> pca(const std::vector<float>& data, size_t numDimensions, size_t& numComponents)
    {
        math::PCA_ALG algorithm = math::PCA_ALG::COV;

        if (data.size() / numDimensions > 20'000)
            algorithm = math::PCA_ALG::SVD;

        math::DATA_NORM norm = math::DATA_NORM::NONE;   // dimension-wise centering is always performed

        std::vector<float> pca;
        bool success = math::pca(data, numDimensions, pca, numComponents, algorithm, norm);

        return { pca, success };
    }

    std::pair<std::vector<float>, bool> pca(const std::vector<float>& data, size_t numDimensions)
    {
        size_t numComps = 2;
        return pca(data, numDimensions, numComps);
    }

    std::pair<std::vector<float>, bool> spectralEmbedding(const GraphBaseInterface& graphView)
    {
        size_t numPoints = static_cast<size_t>(graphView.getNumPoints());
        umappp::NeighborList<int64_t, double> edges(numPoints);

        // transform to umappp data structure
        for (size_t row = 0; row < numPoints; ++row) {
            auto& edgesN = edges[row];
            size_t nn = static_cast<size_t>(graphView.getK(row));
            edgesN.reserve(nn);
            for (size_t k = 1; k < nn; k++) {
                const auto& [id, val] = graphView.getNeighborDistanceN(row, k);
                edgesN.emplace_back(id, val);
            }
        }

        // compute spectral embedding
        const irlba::Options irlba_opt = {};
        const int n_threads = std::thread::hardware_concurrency();
        std::vector<double> emb(numPoints * 2);
        bool success = umappp::normalized_laplacian(edges, 2, emb.data(), irlba_opt, n_threads, /*scale*/ 1.0);

        // convert to float
        auto transformToFloat = [](const std::vector<double>& input) -> std::vector<float> {
            std::vector<float> output(input.size());

            std::transform(input.begin(), input.end(), output.begin(),
                [](double val) { return static_cast<float>(val); });

            return output;
            };

        return { transformToFloat(emb), success };
    }

    std::pair<float, float> randomVec(const float radiusX, const float radiusY) {
        static std::random_device rd;  // Will be used to obtain a seed for the random number engine
        static std::mt19937_64 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        static std::uniform_real_distribution<float> dis(0, 1);

        const float maxR = std::max(radiusX, radiusY);  // sample from a circle - usually radiusX and radiusY are similar

        assert(maxR >= 0);

        const float r = maxR * std::sqrt(dis(gen));     // random radius: uniformly sample from [0, 1], sqrt (important!), then scale to [0, maxR]
        const float t = 2.0f * 3.141592f * dis(gen);    // random angle: uniformly sample from [0, 1] and scale to [0, pi]

        return std::pair{ /* x = */ r * std::cos(t), /* y = */ r * std::sin(t) };
    }


} // namespace sph::utils

