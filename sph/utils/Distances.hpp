#pragma once

#include <hnswlib/hnswlib.h>

#include <cmath>
#include <cstdint>

#if defined(USE_SSE) 
#include <xmmintrin.h>
#endif

namespace sph::utils {

    // If SSE or AVX compiler intrinsic are available
    // use appropriate instruction sets
    inline float L2Sqr(const float* x, const float* y, int64_t d) {

        float(*DIST)(const void*, const void*, const void*);
        DIST = hnswlib::L2Sqr;

#if defined(USE_SSE) 
        if (d % 16 == 0)
#if defined(USE_AVX) 
        DIST = hnswlib::L2SqrSIMD16ExtAVX;
#else
        DIST = hnswlib::L2SqrSIMD16ExtSSE;
#endif
        else if (d % 4 == 0)
            DIST = hnswlib::L2SqrSIMD4Ext;
        else if (d > 16)
            DIST = hnswlib::L2SqrSIMD16ExtResiduals;
        else if (d > 4)
            DIST = hnswlib::L2SqrSIMD4ExtResiduals;
#endif // USE_SSE

        return DIST(x, y, &d);

    }

    inline float L2(const float* x, const float* y, int64_t d)
    {
        return std::sqrt(L2Sqr(x, y, d));
    }

} // namespace sph::utils
