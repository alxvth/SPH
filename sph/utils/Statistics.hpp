#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "CommonDefinitions.hpp"

namespace sph::utils {

    struct GraphInterface;

    struct SparseMatrixStats {
        double sparsity = 0.f;
        double sparsityEffective = 0.f; // considering only effective neighbors (> pruneVale)
        float pruneVale = -1.f;
        size_t nonZeros = 0;
        double averageNonZeros = 0;
        size_t totalEntries = 0;    // assumes square
    };

    void printSparseMatrixStats(const SparseMatrixStats& stats, const std::string& sparseMatrixName = "Sparse matrix");

    SparseMatrixStats sparseMatrixStats(const std::vector<SparseVecSPH>& sparseMatrix, std::optional<float> pruneVal = std::nullopt);

    SparseMatrixStats sparseMatrixStats(const std::vector<SparseVecHDI>& sparseMatrix, std::optional<float> pruneVal = std::nullopt);

    SparseMatrixStats sparseMatrixStats(const GraphInterface* sparseMatrix);

    struct RandomWalkStats {
        uint64_t numWalks = 0;
        std::vector<uint64_t> numHits;  // how many other points where reached starting from i
    };

}
