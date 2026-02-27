#pragma once

#include <algorithm>
#include <cctype>
#include <concepts>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "CommonDefinitions.hpp"

#if !defined(__clang__) && (defined(__GNUC__) || defined(_MSC_VER))
#include <execution>
#if SPH_RELEASE
#define SPH_PARALLEL_EXECUTION std::execution::par,
#else
#define SPH_PARALLEL_EXECUTION std::execution::seq,
#endif
#else // __clang__
#define SPH_PARALLEL_EXECUTION
#endif

namespace sph::utils
{

    /// /////// ///
    /// Sorting ///
    /// /////// ///

    template <typename T>
    concept SortableUniqueContainer = requires(T t) {
        // Check if std::sort and std::unique can be called on T
        { std::sort(std::begin(t), std::end(t)) } -> std::same_as<void>;
        { std::unique(std::begin(t), std::end(t)) } -> std::forward_iterator;
        // Check if T has member function erase
        { t.erase(std::begin(t), std::next(std::begin(t), 1)) } -> std::same_as<typename T::iterator>;
    };

    template <SortableUniqueContainer T>
    void sortAndUnique(T& v)
    {
        if (v.size() <= 1)
            return;

        std::sort(SPH_PARALLEL_EXECUTION
            v.begin(), 
            v.end());
        auto last = std::unique(SPH_PARALLEL_EXECUTION
            v.begin(), 
            v.end());
        v.erase(last, v.end());
    }

    template<typename T, typename U, typename Compare = std::less<T>>
    void synchronizedSort(
        std::span<T> span_a,
        std::span<U> span_b,
        Compare comp = Compare{})
    {
        // Ensure the spans have the same size
        if (span_a.size() != span_b.size()) {
            throw std::invalid_argument("Spans must have equal length");
        }

        // Create an index vector to track original positions
        std::vector<size_t> indices(span_a.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Sort the indices based on the values in span_a using the provided comparator
        std::stable_sort(SPH_PARALLEL_EXECUTION
            indices.begin(), indices.end(),
            [&span_a, &comp](size_t i, size_t j) {
                return comp(span_a[i], span_a[j]);
            }
        );

        // Reorder span_a using the indices
        std::vector<T> sorted_a(span_a.size());
        for (size_t i = 0; i < span_a.size(); ++i) {
            sorted_a[i] = span_a[indices[i]];
        }
        std::copy(sorted_a.begin(), sorted_a.end(), span_a.begin());

        // Reorder span_b using the same indices
        std::vector<U> sorted_b(span_b.size());
        for (size_t i = 0; i < span_b.size(); ++i) {
            sorted_b[i] = span_b[indices[i]];
        }
        std::copy(sorted_b.begin(), sorted_b.end(), span_b.begin());
    }

    /// /////// ///
    /// Finding ///
    /// /////// ///

    template <typename T>
    inline bool contains(std::span<T> sp, const T& x) {
        return std::find(sp.begin(), sp.end(), x) != sp.end();
    }

    inline bool stringContains(const std::string& str, const std::string& substr) {
        return std::search(
            str.begin(), str.end(),
            substr.begin(), substr.end(),
            [](char ch1, char ch2) {
                return std::tolower(ch1) == std::tolower(ch2);
            }
        ) != str.end();
    }

}

