#pragma once

#include "hnswlib/hnswlib.h"    // DISTFUNC

#include "Hierarchy.hpp"
#include "Similarities.hpp"

namespace hnswlib {

    // ---------------
    // Distances based on random walk overlap 
    // ---------------

    // data struct for distance calculation in QFSpace
    struct NeighborWalksSingleOverlapSpaceParameters {
        const sph::utils::Hierarchy* hierarchy = nullptr;

        NeighborWalksSingleOverlapSpaceParameters() = default;

        NeighborWalksSingleOverlapSpaceParameters(const sph::utils::Hierarchy* h) :
            hierarchy(h)
        {
        }
    };

    static float
        DistWalksSingleOverlap(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const sph::utils::ComponentID* cid1 = static_cast<const sph::utils::ComponentID*>(pVect1v);
        const sph::utils::ComponentID* cid2 = static_cast<const sph::utils::ComponentID*>(pVect2v);

        const NeighborWalksSingleOverlapSpaceParameters* sparam = (NeighborWalksSingleOverlapSpaceParameters*)qty_ptr;
        float sim = sph::utils::randomWalksSingleOverlap(sparam->hierarchy->randomWalks, cid1->level, cid1->id, cid2->id);

        // similarity is in [0,1] but we want to return a distance here
        return 1 - sim;
    }


    class NeighborWalksSingleOverlapSpace : public SpaceInterface<float> {

    public:
        NeighborWalksSingleOverlapSpace(const sph::utils::Hierarchy* hierarchy) :
            _fstdistfunc(DistWalksSingleOverlap),
            _data_size(sizeof(sph::utils::ComponentID)),
            _params(hierarchy)
        { }

        ~NeighborWalksSingleOverlapSpace() override {}

        size_t get_data_size() override {
            return _data_size;
        }

        DISTFUNC<float> get_dist_func() override {
            return _fstdistfunc;
        }

        void* get_dist_func_param() override {
            return (void*)&_params;
        }

    private:
        DISTFUNC<float>                             _fstdistfunc = {};
        size_t                                      _data_size = {};
        NeighborWalksSingleOverlapSpaceParameters   _params = {};

    };

}
