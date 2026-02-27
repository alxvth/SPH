#pragma once

#include "hnswlib/hnswlib.h"    // DISTFUNC

#include "CommonDefinitions.hpp"
#include "Data.hpp"
#include "Graph.hpp"
#include "Hierarchy.hpp"
#include "Similarities.hpp"

#include <memory>

namespace hnswlib {

    // ---------------
    // Geodesic path based Neighborhood distance 
    // ---------------

    // data struct for distance calculation in QFSpace
    struct GeodesicPathSpaceParameters {
        const sph::utils::Hierarchy* hierarchy = nullptr;
        const sph::utils::GraphBaseInterface* graph = nullptr;
        const sph::utils::DataView data = {};
        std::shared_ptr<sph::vi64> componentLabels = {};

        GeodesicPathSpaceParameters() = default;

        GeodesicPathSpaceParameters(const sph::utils::Hierarchy* h, const sph::utils::GraphBaseInterface* g, const sph::utils::DataView& d, std::shared_ptr<sph::vi64> componentLabels = nullptr) :
            hierarchy(h), 
            graph(g), 
            data(d),
            componentLabels(componentLabels)
        {
        }
    };

    static float
        DistGeodesicPath(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const sph::utils::ComponentID* cid1 = static_cast<const sph::utils::ComponentID*>(pVect1v);
        const sph::utils::ComponentID* cid2 = static_cast<const sph::utils::ComponentID*>(pVect2v);

        const GeodesicPathSpaceParameters* sparam = (GeodesicPathSpaceParameters*)qty_ptr;

        float dist = sph::utils::geodesicDistance(*(sparam->hierarchy), sparam->graph, sparam->data, *cid1, *cid2, sparam->componentLabels);
        
        return dist;
    }


    class GeodesicPathSpace : public SpaceInterface<float> {

    public:
        GeodesicPathSpace(const sph::utils::Hierarchy* hierarchy, const sph::utils::GraphBaseInterface* graph, const sph::utils::DataView& data, std::shared_ptr<sph::vi64> componentLabels = nullptr) :
            _fstdistfunc(DistGeodesicPath),
            _data_size(sizeof(sph::utils::ComponentID)),
            _params(hierarchy, graph, data, componentLabels)
        { }

        ~GeodesicPathSpace() override {}

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
        GeodesicPathSpaceParameters     _params = {};

    };

}
