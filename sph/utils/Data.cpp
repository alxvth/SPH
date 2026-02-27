#include "Data.hpp"

#include "Logger.hpp"

#include <fmt/core.h>

#include <cassert>
#include <cmath>

namespace sph::utils {

    /// //////// ///
    /// RAW DATA ///
    /// //////// ///

    int64_t DataInterface::checkNumPoints(const int64_t dataSize, const int64_t numDimensions)
    {
        assert(numDimensions >= 0);
        if (numDimensions <= 0)
            Log::error("Data:: numDimensions cannot be <= 0");

        auto dv = std::div(dataSize, numDimensions);

        if (dv.rem != 0)
            Log::error("Data: data size must be integer multiple of dimensions.");

        assert(dv.quot * numDimensions == dataSize);

        return dv.quot;
    }

    Data::Data(const Data& data) :
        dataVec(data.dataVec),
        numDimensions(data.numDimensions),
        numPoints(data.numPoints)
    {
    }

    Data& Data::operator= (const Data& data)
    {
        if (this != &data) {  // Self-assignment check
            dataVec = data.dataVec;
            numDimensions = data.numDimensions;
            numPoints = data.numPoints;
        }
        return *this;
    }

    Data::Data(Data&& data) noexcept :
        dataVec(std::move(data.dataVec)),
        numDimensions(std::move(data.numDimensions)),
        numPoints(std::move(data.numPoints))
    {
    }

    Data::Data(std::vector<float>&& data, const int64_t& numDims) noexcept
    {
        dataVec = std::move(data);
        numDimensions= numDims;
        numPoints = Data::checkNumPoints(dataVec.size(), numDimensions);
    }

    DataView Data::getDataView() const
    {
        return DataView{ &dataVec, &numPoints, &numDimensions };
    }

    Data Data::fromDataView(const DataView dataView) {
        Data out;

        out.dataVec.assign(dataView.getData().begin(), dataView.getData().end());
        out.numDimensions = dataView.getNumDimensions();
        out.numPoints = dataView.getNumPoints();

        return out;
    }

    bool Data::operator==(const Data& other) const
    {
        return dataVec == other.dataVec && numPoints == other.numPoints && numDimensions == other.numDimensions;
    }

    DataView::DataView(const Data& data) :
        dataVec(&data.dataVec),
        numDimensions(&data.numDimensions),
        numPoints(&data.numPoints)
    {
    }

    DataView::DataView(const DataView& dataView) :
        dataVec(dataView.dataVec),
        numDimensions(dataView.numDimensions),
        numPoints(dataView.numPoints)
    {
    }

    DataView& DataView::operator= (const DataView& dataView)
    {
        if (this != &dataView) {  // Self-assignment check
            dataVec = dataView.dataVec;
            numDimensions = dataView.numDimensions;
            numPoints = dataView.numPoints;
        }
        return *this;
    }

    DataView::DataView(const std::vector<float>* data, const int64_t* nPoints, const int64_t* nDims) :
        dataVec(data),
        numDimensions(nDims),
        numPoints(nPoints)
    {
    }

    bool DataView::operator==(const DataView& other) const
    {
        return dataVec == other.dataVec && numPoints == other.numPoints && numDimensions == other.numDimensions;
    }

} // namespace sph::utils
