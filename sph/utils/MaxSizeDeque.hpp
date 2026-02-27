#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <functional>
#include <utility>

namespace sph::utils {

    class SortedMaxSizeDeque {
    public:
        explicit SortedMaxSizeDeque(size_t ms) : maxSize(ms) {
            // Initialize the deque with zeros
            data = std::deque<float>(maxSize, 0.0);
        }

        void insert(float value) {
            // If the smallest element is already greater than or equal to the new value, return
            if (maxSize == 0 || data.back() >= value) {
                return;
            }

            // Check if the deque has reached its maximum size
            if (data.size() >= maxSize) {
                // Remove the smallest element (last element in a sorted deque)
                data.pop_back();
            }

            // Insert the value in a sorted manner
            auto it = std::lower_bound(data.begin(), data.end(), value, std::greater<float>());
            data.insert(it, value);
        }

        const std::deque<float>& getData() const {
            return data;
        }

    private:
        std::deque<float> data;
        size_t maxSize;
    };

    // see https://godbolt.org/z/M84Wjasrb
    // pairs: (value, index) 
    // maximum size of ms, retains ms largest pairs wrt to value
    // 0.9 : 1
    // 0.8 : 5
    // 0.4 : 3
    template <typename T = float, typename S = int64_t>
    class SortedMaxPairSizeDeque {
    public:
        using storageType = std::pair<T, S>;
        using dequeType = std::deque<storageType>;

        SortedMaxPairSizeDeque(size_t ms) : maxSize(ms) {
            // Initialize the deque with zeros
            data = dequeType(maxSize, storageType{ static_cast<T>(0), static_cast<S>(0) });
        }

        void insert(T value, S index) {
            // If the smallest element is already greater than or equal to the new value, return
            if (maxSize == 0 || data.back().first >= value) {
                return;
            }

            // Check if the deque has reached its maximum size
            if (data.size() >= maxSize) {
                // Remove the smallest element (last element in a sorted deque)
                data.pop_back();
            }

            // Insert the value in a sorted manner
            auto comp = [](storageType& dataPair, T value) {
                return dataPair.first > value;
                };
            auto it = std::lower_bound(data.begin(), data.end(), value, comp);
            data.insert(it, storageType{ value, index });
        }

        const dequeType& getData() const {
            return data;
        }

    private:
        dequeType data;
        size_t maxSize;
    };

    // see https://godbolt.org/z/1of6xn7rn
    // pairs: (index, value) sorts/retains by value
    // maximum size of ms, retains ms largest pairs wrt to value
    // 1 : 0.9
    // 5 : 0.8
    // 3 : 0.4
    // the internal storage (a deque) keeps all values sorted based on T from large to small
    template <typename S = int64_t, typename T = float>
    class SortedMaxPairSizeDequeR {
    public:
        using storageType = std::pair<S, T>;
        using dequeType = std::deque<storageType>;
        static constexpr T DEFAULT = std::numeric_limits<T>::min();

        SortedMaxPairSizeDequeR(size_t ms) : maxSize(ms) {
            // Initialize the deque with zeros
            data = dequeType(maxSize, storageType{ static_cast<S>(0), DEFAULT });
        }

        void insert(S index, T value) {
            // If the smallest element is already greater than or equal to the new value, return
            if (maxSize == 0 || data.back().second >= value) {  // Compare with .second now
                return;
            }

            // Check if the deque has reached its maximum size
            if (data.size() >= maxSize) {
                // Remove the smallest element (last element in a sorted deque)
                data.pop_back();
            }

            // Insert the value in a sorted manner
            auto comp = [](const storageType& dataPair, const T value) {
                return dataPair.second > value;  // Compare with .second for sorting
                };
            auto it = std::lower_bound(data.begin(), data.end(), value, comp);
            data.insert(it, storageType{ index, value });  // Swapped order in construction
        }

        void prune() {
            data.erase(
                std::remove_if(
                    data.begin(), data.end(),
                    [](const storageType& entry) { return entry.second == DEFAULT; }),
                data.end());
        }

        const dequeType& getData() const {
            return data;
        }

    private:
        dequeType data;
        size_t maxSize;
    };

    // see https://godbolt.org/z/1of6xn7rn
    // pairs: (index, value) sorts/retains by value
    // maximum size of ms, retains ms smallest pairs wrt to value
    // 4 0.1
    // 0 0.3
    // 5 0.4
    // the internal storage (a deque) keeps all values sorted based on T from small to large
    template <typename S = int64_t, typename T = float>
    class SortedMinPairSizeDequeR {
    public:
        using storageType = std::pair<S, T>;
        using dequeType = std::deque<storageType>;
        static constexpr T DEFAULT = std::numeric_limits<T>::max();

        SortedMinPairSizeDequeR(size_t ms) : maxSize(ms) {
            // Initialize the deque with zeros
            data = dequeType(maxSize, storageType{ static_cast<S>(0), DEFAULT });
        }

        void insert(S index, T value) {
            // If the largest element is already smaller than or equal to the new value, return
            if (maxSize == 0 || data.back().second <= value) {  // Compare with .second now
                return;
            }

            // Check if the deque has reached its maximum size
            if (data.size() >= maxSize) {
                // Remove the smallest element (last element in a sorted deque)
                data.pop_back();
            }

            // Insert the value in a sorted manner
            auto comp = [](const storageType& dataPair, const T value) {
                return dataPair.second < value;  // Compare with .second for sorting
                };
            auto it = std::lower_bound(data.begin(), data.end(), value, comp);
            data.insert(it, storageType{ index, value });  // Swapped order in construction
        }

        void prune() {
            data.erase(
                std::remove_if(
                    data.begin(), data.end(),
                    [](const storageType& entry) { return entry.second == DEFAULT; }),
                data.end());
        }

        const dequeType& getData() const {
            return data;
        }

    private:
        dequeType data;
        size_t maxSize;
    };


} // namespace sph::utils
