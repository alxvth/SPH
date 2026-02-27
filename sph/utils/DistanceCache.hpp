#pragma once

#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ankerl/unordered_dense.h>

namespace sph::utils {

    // Helper function to split the tuple into two halves
    template<std::size_t Start, std::size_t End, typename Tuple>
    auto _sub_tuple(Tuple&& tuple) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(std::get<I + Start>(std::forward<Tuple>(tuple))...);
        }(std::make_index_sequence<End - Start>{});
    }

    // Main function to reorder the tuple
    template<typename Tuple>
    auto _reorder_tuple(Tuple&& tuple) {
        constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;
        constexpr std::size_t Half = N / 2;
        
        // Extract the second half and first half, then concatenate them
        auto second_half = _sub_tuple<Half, N>(std::forward<Tuple>(tuple));
        auto first_half = _sub_tuple<0, Half>(std::forward<Tuple>(tuple));
        
        return std::tuple_cat(std::move(second_half), std::move(first_half));
    }

    // Helper function to generate a tuple with N elements of type T
    template <typename T, std::size_t... Is>
    auto _make_tuple_of_size_impl(std::index_sequence<Is...>, const T& value) {
        return std::make_tuple(((void)Is, value)...); // Creates a tuple of N elements, all set to `value`
    }

    // DistanceCache<int64_t, 2, float> for distance where IDs are one int64_t respectively
    // DistanceCache<int64_t, 4, float> for distance where IDs are two int64_t respectively
    template<typename KEY_TYPE, std::size_t KEY_LENGTH, typename VALUE>
    struct DistanceCache
    {
        using key      = decltype(_make_tuple_of_size_impl(std::make_index_sequence<KEY_LENGTH>{}, std::declval<KEY_TYPE>()));
        using hash     = ankerl::unordered_dense::hash<key>;
        using hashmap  = ankerl::unordered_dense::map<key, VALUE, hash>;

        static_assert(std::tuple_size_v<key> % 2 == 0, "KEY_LENGTH must be even");

        hashmap         _cacheDistance = {};
        std::mutex     _mtx = {};
        bool           _useSymmetricLookup = true;
        bool           _useCache = false;

        void setUseCacheDistanceCache(bool c) {
            _useCache = c;
        }

        void setUseSymmetricLookupDistanceCache(bool c) {
            _useSymmetricLookup = c;
        }

        bool getUseCacheDistanceCache() { 
            return _useCache; 
        }

        bool getUseSymmetricLookupDistanceCache() { 
            return _useSymmetricLookup; 
        }

        void clearCacheDistanceCache() {
            _cacheDistance.clear();
        }

        void add(const key& k, float dist) {
            std::lock_guard<std::mutex> lock(_mtx);
            _cacheDistance[k] = dist;
        }

        void add(key&& k, float dist) {
            add(k, dist);
        }

        bool contains(const key& k, float& dist) {
            std::lock_guard<std::mutex> lock(_mtx);

            auto lookUp = [&](const key& k, float& dist) -> bool {
                if (!_cacheDistance.contains(k))
                    return false;

                dist = _cacheDistance[k];
                return true;
            };

            if(lookUp(k, dist))
                return true;

            if (_useSymmetricLookup && lookUp(_reorder_tuple(k), dist))
                return true;

            return false;
        }

        bool contains(key&& k, float& dist) {
            return contains(k, dist);
        }

        void reserve(size_t capacity) {
            _cacheDistance.reserve(capacity);
        }


    }; // struct DistanceCache

} // namespace sph::utils
