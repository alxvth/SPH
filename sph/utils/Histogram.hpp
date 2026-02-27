#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sph::utils {

    /*! Base class for histograms
    */
    template <class outType, class inType>
    class HistogramBase
    {
    public:
        using EigenVectorOut = Eigen::Vector<outType, -1>;

    public:
        HistogramBase() = delete;
        HistogramBase(inType min, inType max, uint64_t numberOfBins);
        // The last bin might be smaller than the rest if (max-min)/binWidth does not yield an integer
        HistogramBase(inType min, inType max, double binWidth);

        void fill(const inType value);
        void fill(const std::vector<inType>& values);

        void fill(const inType value, outType weight);
        void fill(const std::vector<inType>& values, const std::vector<outType>& weights);

    public: // Getter
        uint64_t getNumBins() const { return _counts.size(); };
        uint64_t getCount(uint64_t bin) const { return _counts[bin]; };

        uint64_t getCount() const { return _countBinValid; };
        uint64_t getCountAll() const { return _countBinTotal; };
        uint64_t getCountUnderflow() const { return _countBinUnderflow; };
        uint64_t getCountOverflow() const { return _countBinOverflow; };

        inType getMin() const { return _minVal; };
        inType getMax() const { return _maxVal; };
        float getBinLower(uint64_t bin) const { return _minVal + bin * _binWidth; };
        float getBinUpper(uint64_t bin) const { return _minVal + (bin + 1) * _binWidth; };

        auto cbegin() const { return _counts.cbegin(); };
        auto cend() const { return _counts.cend(); };

        outType operator[](size_t index) const;

        const EigenVectorOut& counts() const { return _counts; };
        Eigen::VectorXd normalizedCounts() const { return _counts.template cast<double>() / _counts.sum(); };

    private:
        EigenVectorOut  _counts{};
        uint64_t        _countBinOverflow{ 0 };
        uint64_t        _countBinUnderflow{ 0 };
        uint64_t        _countBinTotal{ 0 };
        uint64_t        _countBinValid{ 0 };
        double          _binWidth{ 0 };
        double          _binNormed{ 0 };
        inType          _minVal{ 0 };
        inType          _maxVal{ 0 };
        uint64_t        _numBins{ 0 };
    };

    template< class outType, class inType>
    HistogramBase<outType, inType>::HistogramBase(inType min, inType max, uint64_t numberOfBins) :
        _minVal(min),
        _maxVal(max),
        _numBins(numberOfBins)
    {
        assert(min <= max);

        _binWidth = (_maxVal - _minVal) / static_cast<double>(_numBins);

        _counts = EigenVectorOut::Zero(_numBins);
        _binNormed = (float)_numBins / (_maxVal - _minVal);
    }

    template< class outType, class inType>
    HistogramBase<outType, inType>::HistogramBase(inType min, inType max, double binWidth) :
        _minVal(min),
        _maxVal(max),
        _binWidth(binWidth)
    {
        _numBins = static_cast<uint64_t>(std::ceil((_maxVal - _minVal) / static_cast<double>(_binWidth)));

        _counts = EigenVectorOut::Zero(_numBins);
        _binNormed = (float)_numBins / (_maxVal - _minVal);
    }

    template< class outType, class inType>
    void HistogramBase<outType, inType>::fill(const inType value, const outType weight) {
        uint64_t binID;
        if (value >= _minVal && value < _maxVal) {
            binID = static_cast<uint64_t> (std::floor((value - _minVal) * _binNormed));
            _counts[binID] += weight;
            _countBinValid++;
        }
        else if (value == _maxVal)
        {
            _counts[_numBins - 1] += weight;
            _countBinValid++;
        }
        else if (value > _maxVal) {
            _countBinOverflow++;
        }
        else {
            _countBinUnderflow++;
        }

        _countBinTotal++;
    }

    template< class outType, class inType>
    void HistogramBase< outType, inType>::fill(const inType value) {
        fill(value, static_cast< outType>(1));
    }

    template< class outType, class inType>
    void HistogramBase< outType, inType>::fill(const std::vector<inType>& values) {
        for (const float& value : values)
            fill(value);
    }

    template< class outType, class inType>
    void HistogramBase< outType, inType>::fill(const std::vector<inType>& values, const std::vector<outType>& weights) {
        assert(values.size() == weights.size());

        for (uint64_t i = 0; i < values.size(); i++)
            fill(values[i], weights[i]);
    }

    template< class outType, class inType>
    outType HistogramBase< outType, inType>::operator[](size_t index) const {
        assert(index >= 0 && index < _numBins);
        return _counts[index];
    }


    /*! Histogram class
     *
     * If newVal == binMax then it will not count as overflow but is counted in the largest bin
     */
    template <class inType = float>
    class Histogram : public HistogramBase<uint64_t, inType>
    {
    public:
        Histogram() = delete;
        Histogram(inType min, inType max, uint64_t numberOfBins) : HistogramBase<uint64_t, inType>(min, max, numberOfBins) { };
        Histogram(inType min, inType max, double binWidth) : HistogramBase<uint64_t, inType>(min, max, binWidth) { };
    };

} // namespace sph::utils
