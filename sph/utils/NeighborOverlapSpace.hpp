#pragma once

#include "hnswlib/hnswlib.h"    // DISTFUNC

#include "Graph.hpp"
#include "Hierarchy.hpp"
#include "Similarities.hpp"

#include <Eigen/SparseCore>

namespace hnswlib {

    // ---------------
    // kNN Overlap based Neighborhood distance 
    // ---------------

    // data struct for distance calculation in QFSpace
    struct NeighborOverlapSpaceParameters {
        const sph::utils::Hierarchy* hierarchy = nullptr;
        const sph::utils::GraphBaseInterface* graph = nullptr;

        NeighborOverlapSpaceParameters() = default;

        NeighborOverlapSpaceParameters(const sph::utils::Hierarchy* h, const sph::utils::GraphBaseInterface* g) :
            hierarchy(h), 
            graph(g) 
        {
        }
    };

    static float
        DistNeighborOverlap(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const sph::utils::ComponentID* cid1 = static_cast<const sph::utils::ComponentID*>(pVect1v);
        const sph::utils::ComponentID* cid2 = static_cast<const sph::utils::ComponentID*>(pVect2v);

        const NeighborOverlapSpaceParameters* sparam = (NeighborOverlapSpaceParameters*)qty_ptr;

        float sim = sph::utils::simNeighborOverlap(*(sparam->hierarchy), sparam->graph, *cid1, *cid2);
        
        // similarity is in [0,1] but we want to return a distance here
        return 1.f - sim;
    }


    class NeighborOverlapSpace : public SpaceInterface<float> {

    public:
        NeighborOverlapSpace(const sph::utils::Hierarchy* hierarchy, const sph::utils::GraphBaseInterface* graph) :
            _fstdistfunc(DistNeighborOverlap),
            _data_size(sizeof(sph::utils::ComponentID)),
            _params(hierarchy, graph)
        { }

        ~NeighborOverlapSpace() override {}

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
        DISTFUNC<float>                 _fstdistfunc = {};
        size_t                          _data_size = {};
        NeighborOverlapSpaceParameters  _params = {};

    };

}
