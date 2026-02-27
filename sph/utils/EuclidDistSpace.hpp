#pragma once

#include "hnswlib/hnswlib.h"    // DISTFUNC

#include "Data.hpp"
#include "Hierarchy.hpp"        // ComponentID
#include "Similarities.hpp"

namespace hnswlib {

    // ---------------
    // Euclidean distance based Neighborhood distance 
    // ---------------

    // data struct for distance calculation in QFSpace
    struct EuclidDistSpaceParameters {
        const sph::utils::Hierarchy* hierarchy = nullptr;
        const sph::utils::DataView data = {};

        EuclidDistSpaceParameters() = default;

        EuclidDistSpaceParameters(const sph::utils::Hierarchy* h, const sph::utils::DataView& d) :
            hierarchy(h), 
            data(d)
        {
        }
    };

    static float
        DistEuclid(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const sph::utils::ComponentID* cid1 = static_cast<const sph::utils::ComponentID*>(pVect1v);
        const sph::utils::ComponentID* cid2 = static_cast<const sph::utils::ComponentID*>(pVect2v);

        const EuclidDistSpaceParameters* sparam = (EuclidDistSpaceParameters*)qty_ptr;

        float dist = sph::utils::euclidDistance(*(sparam->hierarchy), sparam->data, *cid1, *cid2);
        
        return dist;
    }


    class EuclidDistSpace : public SpaceInterface<float> {

    public:
        EuclidDistSpace(const sph::utils::Hierarchy* hierarchy, const sph::utils::DataView& data) :
            _fstdistfunc(DistEuclid),
            _data_size(sizeof(sph::utils::ComponentID)),
            _params(hierarchy, data)
        { }

        ~EuclidDistSpace() override {}

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
        DISTFUNC<float>             _fstdistfunc = {};
        size_t                      _data_size = {};
        EuclidDistSpaceParameters   _params = {};

    };

}
