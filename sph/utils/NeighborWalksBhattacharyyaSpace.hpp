#pragma once

#include "hnswlib/hnswlib.h"    // DISTFUNC

#include "Hierarchy.hpp"
#include "Similarities.hpp"

#include <Eigen/SparseCore>

namespace hnswlib {

    // ---------------
    // Distances based on random walk Bhattacharyya 
    // ---------------

    // data struct for distance calculation in QFSpace
    struct NeighborWalksBhattacharyyaSpaceParameters {
        const sph::utils::Hierarchy* hierarchy = nullptr;

        NeighborWalksBhattacharyyaSpaceParameters() = default;

        NeighborWalksBhattacharyyaSpaceParameters(const sph::utils::Hierarchy* h) :
            hierarchy(h)
        {
        }
    };

    static float
        DistWalksBhattacharyya(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const sph::utils::ComponentID* cid1 = static_cast<const sph::utils::ComponentID*>(pVect1v);
        const sph::utils::ComponentID* cid2 = static_cast<const sph::utils::ComponentID*>(pVect2v);

        assert(cid1->level == cid2->level);

        const NeighborWalksBhattacharyyaSpaceParameters* sparam = (NeighborWalksBhattacharyyaSpaceParameters*)qty_ptr;
        float sim = sph::utils::randomWalksBhattacharyya(sparam->hierarchy->randomWalks, cid1->level, cid1->id, cid2->id);

        // similarity is in [0,1] but we want to return a distance here
        return 1 - sim;
    }


    class NeighborWalksBhattacharyyaSpace : public SpaceInterface<float> {

    public:
        NeighborWalksBhattacharyyaSpace(const sph::utils::Hierarchy* hierarchy) :
            _fstdistfunc(DistWalksBhattacharyya),
            _data_size(sizeof(sph::utils::ComponentID)),
            _params(hierarchy)
        { }

        ~NeighborWalksBhattacharyyaSpace() override {}

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
        NeighborWalksBhattacharyyaSpaceParameters   _params = {};

    };

}
