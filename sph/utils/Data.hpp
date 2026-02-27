#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace sph::utils {

    struct Data;
    struct DataView;

    struct DataInterface
    {
        virtual const float* data() const = 0;

        virtual const std::vector<float>& getData() const = 0;
        virtual std::vector<float>& getData() = 0;
        virtual int64_t getNumPoints() const = 0;
        virtual int64_t getNumDimensions() const = 0;

        virtual inline std::span<const float> getValuesAt(uint64_t id) const = 0;

        // returns number of data points, warn if dataSize / numDimensions is not a natural number
        static int64_t checkNumPoints(const int64_t dataSize, const int64_t numDimensions);

        virtual bool isValid() const = 0;

        virtual ~DataInterface() = default;

        virtual float operator[](size_t index) const = 0;
    };

    struct Data : DataInterface
    {
        Data() = default;
        Data(const Data& data);
        Data& operator= (const Data& data);
        Data(Data&& data) noexcept;
        Data(std::vector<float>&& data, const int64_t& numDims) noexcept;

        std::vector<float>  dataVec = {};
        int64_t             numDimensions = 0;
        int64_t             numPoints = 0;

        DataView getDataView() const;

        const float* data() const override { return dataVec.data(); };
        const std::vector<float>& getData() const override { return dataVec; };
        std::vector<float>& getData() override { return dataVec; };
        int64_t getNumPoints() const override { return numPoints; };
        int64_t getNumDimensions() const override { return numDimensions; };

        inline std::span<const float> getValuesAt(uint64_t id) const override {
            return std::span{ dataVec.data() + id * numDimensions, dataVec.data() + id * numDimensions + numDimensions };
        };

        inline std::span<float> getValuesAtRef(uint64_t id) {
            return std::span{ dataVec.data() + id * numDimensions, dataVec.data() + id * numDimensions + numDimensions };
        };

        bool isValid() const override {
            return (dataVec.size() > 0) && (dataVec.size() == static_cast<size_t>(numPoints * numDimensions));
        }

        virtual inline void clear() {
            dataVec.clear();
            numDimensions = 0;
            numPoints = 0;
        }

        bool operator==(const Data& other) const;

        float& operator[](size_t index) {
            return dataVec[index];
        }

        float operator[](size_t index) const override {
            return dataVec[index];
        }

        static Data fromDataView(const DataView dataView);
    };

    struct DataView : DataInterface
    {
        DataView() = default;
        DataView(const Data& data);
        DataView(const DataView& dataView);
        DataView& operator= (const DataView& dataView);
        DataView(const std::vector<float>* data, const int64_t* numPoints, const int64_t* numDims);

        const std::vector<float>* dataVec = nullptr;
        const int64_t* numDimensions = nullptr;
        const int64_t* numPoints = nullptr;

        const float* data() const override { return dataVec->data(); };
        const std::vector<float>& getData() const override { return *dataVec; };
        std::vector<float>& getData() override { return *const_cast<std::vector<float>*>(dataVec); };
        int64_t getNumPoints() const override { return *numPoints; };
        int64_t getNumDimensions() const override { return *numDimensions; };

        inline std::span<const float> getValuesAt(uint64_t id) const override {
            return std::span{ dataVec->data() + id * *numDimensions, dataVec->data() + id * *numDimensions + *numDimensions };
        };

        bool isValid() const override {
            return (dataVec != nullptr) && (dataVec->size() > 0) && (dataVec->size() == static_cast<size_t>(*numPoints * *numDimensions));
        }

        bool operator==(const DataView& other) const;

        float operator[](size_t index) const override {
            return (*dataVec)[index];
        }
    };

} // namespace sph::utils
