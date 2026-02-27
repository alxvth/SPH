#pragma once

#include "Data.hpp"
#include "Settings.hpp"

namespace sph::utils {

    // Scales data in-place
    void scale(Data& data, const Scaler scaler);

    // Returns new, scaled data
    inline Data scale(const DataView dataView, const Scaler scaler)
    {
        Data outScaledData = Data::fromDataView(dataView);
        scale(outScaledData, scaler);
        return outScaledData;
    }

} // sph::utils