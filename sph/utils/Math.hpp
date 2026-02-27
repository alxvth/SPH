#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Algorithms.hpp"
#include "CommonDefinitions.hpp"
#include "Graph.hpp"

namespace Eigen
{
    // need to include <Eigen/Dense>
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    class Matrix;
    typedef Matrix<float, -1, -1, 0, -1, -1> MatrixXf;
}

namespace sph::utils {

    /// /////////// ///
    ///   General   ///
    /// /////////// ///
    template <typename T>
    std::vector<T> computeMean(const std::vector<T>& points, const size_t numDims) {
        const size_t numPoints = points.size() / numDims;
        std::vector<double> means(numDims);

        for (size_t i = 0; i < numPoints; ++i) {
            for (size_t dim = 0; dim < numDims; ++dim) {
                means[dim] += static_cast<double>(points[numDims * i + dim]);
            }
        }

        for (auto& mean : means) {
            mean /= static_cast<double>(numPoints);
        }

        std::vector<T> res(numDims);
        std::transform(means.begin(), means.end(), res.begin(),
            [](double value) { return static_cast<T>(value); });

        return res;
    }

    template <typename T>
    std::vector<T> computeStd(const std::vector<T>& points, const std::vector<T>& means) {
        const size_t numDims = means.size();
        const size_t numPoints = points.size() / numDims;
        std::vector<double> stds(numDims);

        for (size_t i = 0; i < numPoints; ++i) {
            for (size_t dim = 0; dim < numDims; ++dim) {
                stds[dim] += std::pow(static_cast<double>(points[numDims * i + dim]) - static_cast<double>(means[dim]), 2.0);
            }
        }

        for (auto& std : stds) {
            std = std::sqrt(std / static_cast<double>(numPoints));
        }

        std::vector<T> res(numDims);
        std::transform(stds.begin(), stds.end(), res.begin(),
            [](double value) { return static_cast<T>(value); });

        return res;
    }

    /// ///////////// ///
    ///   Distances   ///
    /// ///////////// ///

    // Logistic function 1 / (1 + exp(x)), maps [0, inf] to[0.5, 1]
    [[nodiscard]] static inline float logfunc(float x)
    {
        return 1.0f / (1.0f + expf(-x));
    };

    // Use logistic function which maps [0, inf) to[0.5, 1), which we then adjust this to[0, 1]
    [[nodiscard]] static inline float sigmoid(float x)
    {
        return (logfunc(x) * 2.0f) - 1.0f;
    };

    // inversion/reciprocal, maps [0, inf) to [1, 0) with 1 / (1 + x)
    [[nodiscard]] static inline float invlin(float x)
    {
        assert(x != -1.f);
        return 1.0f / (1.0f + x);
    };

    // inversion/reciprocal, maps [0, inf] to [1, 0] with 1 / (1 + x)
    [[nodiscard]] static inline float inv(float x)
    {
        assert(x != 0.f);
        return 1.0f / x;
    };

    int64_t randomNumberUniform(int64_t min, int64_t max);

    float jaccardCoeff(const SparseVecSPH& A, const SparseVecSPH& B);

    float jaccardCoeff(const std::vector<float>& a, const std::vector<float>& b);

    void clampValues(std::vector<float>& data, float newMin, float newMax);

    inline void clampDistances(GraphBaseInterface& graphView, float newMin, float newMax)
    {
        auto& knnDistances = graphView.getKnnDistances();
        clampValues(knnDistances, newMin, newMax);
    }

    // interpolation: 0 midpoint, 1 linear
    float computeQuantile(const std::vector<float>& data, const float quantile, const std::vector<float>& ignoreVals = {}, const int interpolation = 0);

    float symmetricHausdorffDistance(const Eigen::MatrixXf& distanceMatrix);

    /// ///////////// ///
    /// Normalization ///
    /// ///////////// ///

    // Such that sum of all values [between (inclusive) begin and (exclusive) end iterators] == 1
    // L1 Normalization
    template<std::forward_iterator Ite, std::sentinel_for<Ite> Sen>
    void normalizeUnit(Ite begin, Sen end)
    {
        double sum = std::accumulate(begin, end, 0.0);

        std::transform(SPH_PARALLEL_EXECUTION
            begin, end, // from, to
            begin,      // write to the same location
            [sum](auto value) { return static_cast<decltype(value)>(value / sum); });
    }

    // Such that all values are in [-1 , 1]
    template<std::forward_iterator Ite, std::sentinel_for<Ite> Sen>
    void normalizeScale(Ite begin, Sen end)
    {
        auto max = std::abs(*std::max_element(begin, end));

        std::for_each(SPH_PARALLEL_EXECUTION
            begin, 
            end, 
            [max](auto& val) { val /= max; });
    }

    // Such that all values are in [0, 1] -> subtract min
    template<std::forward_iterator Ite, std::sentinel_for<Ite> Sen>
    void normalizeMinMax(Ite begin, Sen end)
    {
      using val_t = std::iterator_traits<Ite>::value_type;
      const auto [minIt, maxIt] = std::minmax_element(begin, end);

      val_t max = *maxIt;
      val_t min = *minIt;
      val_t range = max - min;

      assert(range >= 0);

      if (range == 0)
        range = 1;

      std::transform(SPH_PARALLEL_EXECUTION
        begin, end, // from, to
        begin,      // write to the same location
        [min, range](val_t val) { return (val - min) / range; });
    }

    // Such that all values are in (0, 1]
    template<std::forward_iterator Ite, std::sentinel_for<Ite> Sen>
    void normalizeUniform(Ite begin, Sen end)
    {
      using val_t = std::iterator_traits<Ite>::value_type;
      const auto [minIt, maxIt] = std::minmax_element(begin, end);

      val_t max = *maxIt;

      std::transform(SPH_PARALLEL_EXECUTION
        begin, end, // from, to
        begin,      // write to the same location
        [max](val_t val) { return val / max; });
    }

    // Such that all values are in [0, 1], channel-wise, first subtract min
    // see https://godbolt.org/z/dsdfcWe6W
    void normalizeData(std::vector<float>& data, size_t numDims);

    // Such that all values are in (0, 1] channel-wise
    // see https://godbolt.org/z/9E7Mxoxaa
    void normalizeChannelsUniform(std::vector<float>& data, size_t numDims);

    // Computes z = (x - u) / s where u is the mean and s is the standard deviation
    // I.e. centers data around mean with unit standard deviation using, as in
    // https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    void normalizeStandard(std::vector<float>& data, size_t numDims);

    /// //////////// ///
    ///     Misc    ///
    /// //////////// ///

    std::pair<std::vector<float>, bool> pca(const std::vector<float>& data, size_t numDimensions, size_t& numComponents);

    // returns first two principal components
    // best call normalizeStandard(data, numDims) before
    std::pair<std::vector<float>, bool> pca(const std::vector<float>& data, size_t numDimensions);

    std::pair<std::vector<float>, bool> spectralEmbedding(const GraphBaseInterface& graphView);

    template<typename T>
    std::vector<size_t> findUniques(const std::vector<T>& nums) {
        // Step 1: Count the occurrences of each number
        std::unordered_map<size_t, T> counts;
        for (const auto& num : nums) {
            ++counts[num];
        }

        // Step 2: Collect numbers that occur exactly once
        std::vector<size_t> unique_nums;
        for (const auto& [num, count] : counts) {
            if (count == 1) {
                unique_nums.push_back(num);
            }
        }

        return unique_nums;
    }

    // Generates combinations 
    // n=2 : (0, 1)
    // n=3 : (0, 1), (0, 2), (1, 2)
    // n=4 : (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    template<typename T>
    std::vector<std::pair<T, T>> generateCombinations(const T& n) {
        std::vector<std::pair<T, T>> combinations;
        
        for (T i = 0; i < n; ++i) {
            for (T j = i + 1; j < n; ++j) {
                combinations.emplace_back(i, j);
            }
        }
        
        return combinations;
    }

    std::pair<float, float> randomVec(const float radiusX, const float radiusY);

    template<typename T = float>
    inline bool isBasicallyZero(T num, double eps = 0.000001)
    {
        return static_cast<double>(std::abs(num)) < eps;
    }

    template<typename T = float, typename S = float>
    inline bool isBasicallyEqual(T a, S b, double eps = 0.000001)
    {
        return std::abs(static_cast<double>(a) - static_cast<double>(b)) < eps;
    }

} // namespace sph::utils

