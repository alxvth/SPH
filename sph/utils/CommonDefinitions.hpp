#pragma once

#include <cstdint>
#include <vector>

/// ///////////// ///
///    Macros     ///
/// ///////////// ///

#ifdef NDEBUG
#define SPH_PARALLEL _Pragma("omp parallel for")
#else
#define SPH_PARALLEL
#endif

#define SPH_PARALLEL_ALWAYS _Pragma("omp parallel for")

#ifdef NDEBUG
#define SPH_PARALLEL_DYNAMIC _Pragma("omp parallel for schedule(dynamic)")
#else
#define SPH_PARALLEL_DYNAMIC
#endif

#ifdef NDEBUG
#define SPH_PARALLEL_THREADS_PRAGMA(X) _Pragma(#X)
#define SPH_PARALLEL_THREADS(numThreads) SPH_PARALLEL_THREADS_PRAGMA(omp parallel for num_threads(numThreads))
#else
#define SPH_PARALLEL_THREADS(numThreads)
#endif

#ifdef NDEBUG
#define SPH_PARALLEL_DYNAMIC_THREADS_PRAGMA(X) _Pragma(#X)
#define SPH_PARALLEL_DYNAMIC_THREADS(numThreads) SPH_PARALLEL_THREADS_PRAGMA(omp parallel for schedule(dynamic) num_threads(numThreads))
#else
#define SPH_PARALLEL_DYNAMIC_THREADS(numThreads)
#endif

#ifdef NDEBUG
#define SPH_PARALLEL_CRITICAL _Pragma("omp critical ")
#else
#define SPH_PARALLEL_CRITICAL
#endif

#ifdef NDEBUG
#define SPH_DEBUG 0
#define SPH_RELEASE 1
#else
#define SPH_DEBUG 1
#define SPH_RELEASE 0
#endif


#if defined(__aarch64__) || \
    defined(_M_ARM64) || \
    defined(__ARM64_ARCH_8A__) || \
    defined(__ARM_ARCH_ISA_A64) || \
    (defined(__ARM_ARCH) && __ARM_ARCH >= 8) || \
    defined(__arm64) || \
    defined(__arm64__)
#define IS_ARM64 1
#endif

/// //////////// ///
///    Types     ///
/// //////////// ///

// Forward declaration of HDILib template classes
namespace hdi
{
    namespace data
    {
        // need to include <hdi/data/map_mem_eff.h>
        template <typename T1, typename T2>
        class MapMemEff;
    }
}

// Forward declaration of Eigen template classes
namespace Eigen
{
    // need to include <Eigen/SparseCore>
    template <typename Scalar, int Options, typename Index>
    class SparseVector;
}

namespace sph {
    using SparseVecHDI = ::hdi::data::MapMemEff<uint32_t, float>;
    using SparseMatHDI = std::vector<SparseVecHDI>;
    using vSparseMatHDI = std::vector<SparseMatHDI>;

    using SparseVecSPH = Eigen::SparseVector<float, 0, int32_t>; // Eigen::ColMajor
    using SparseMatSPH = std::vector<SparseVecSPH>;
    using vSparseMatSPH = std::vector<SparseMatSPH>;

    using vvui64 = std::vector<std::vector<uint64_t>>;
    using vvi64 = std::vector<std::vector<int64_t>>;
    using vvf64 = std::vector<std::vector<double>>;
    using vvf32 = std::vector<std::vector<float>>;

    using vui64 = std::vector<uint64_t>;
    using vi64 = std::vector<int64_t>;
    using vf64 = std::vector<double>;
    using vf32 = std::vector<float>;

} // namespace sph
