#pragma once

#include "Data.hpp"
#include "Graph.hpp"

#include <optional>
#include <vector>

namespace sph::utils {
    void computeButeForce(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph);
    void computeIndexFlat(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph);
    void computeIndexIVFFlat(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph);
    void computeIndexHNSW(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph);
    void computeIndexHNSWSQ(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph);
    void computeIndexHNSW_IVFPQ(int64_t nn, int faissMetric, const utils::DataView& data, utils::Graph& knnGraph);

    std::optional<std::vector<int64_t>> checkAllNeighborsExist(const utils::GraphView knnGraphView);

} // namespace sph::utils
