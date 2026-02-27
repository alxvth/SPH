#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "CommonDefinitions.hpp"
#include "Settings.hpp"

#include <Eigen/Dense>          // Eigen::MatrixXf
#include <Eigen/SparseCore>     // Eigen::SparseVector

namespace sph::utils {

    struct Graph;
    struct GraphInterface;
    struct GraphBaseInterface;
    struct GraphView;
    struct SparseMatrixStats;

    /// ////////////// ///
    ///  Normalization ///
    /// ////////////// ///

    // Such that all values sum to 1
    void normalizeSparseMatrix(SparseMatSPH& sparseMat);

    // Use the given weights
    void normalizeSparseMatrixWith(SparseMatSPH& sparseMat, std::vector<uint64_t> rowWeights);

    // Divide each vector enty by normVal
    template<typename T>
    void normalizeSparseVectorWith(SparseVecSPH& sparseVec, T normVal)
    {
        for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
            it.valueRef() /= normVal;
    }

    // Such that sum of all values == 1
    void normalizeUnitSparseVector(SparseVecSPH& sparseVec);

    // Such that sum of all values in each row == 1
    void normalizeUnitSparseMatrix(SparseMatSPH& sparseMat, bool verbose = true);

    // Such all values are in [0, 1]
    void normalizeMinMaxSparseVector(SparseVecSPH& sparseVec);

    // Defaults to normalizeUnitSparseVector, such that sum of all values == 1
    inline void normalizeSparseVector(SparseVecSPH& sparseVec)
    {
        normalizeUnitSparseVector(sparseVec);
    }

    /// ////////////// ///
    ///  Random Walks  ///
    /// ////////////// ///

    inline float stepLinear(uint64_t step, uint64_t walkLength) {
        return static_cast<float>(1.0 - (static_cast<double>(step) / walkLength));
    }

    inline float normWeight(float index) {
        return std::exp(-0.5f * std::pow(index, 2.f));
    }

    // 3.f to cover three sigma, see https://godbolt.org/z/z1f4jsf4d
    inline float stepNormal(uint64_t step, uint64_t walkLength) {
        return normWeight(step * 3.f / walkLength);
    }

    void doRandomWalks(const SparseMatSPH& similarities, const RandomWalkSettings& settings, SparseMatSPH& randomWalks, SparseMatrixStats& stats, bool verbose = false);

    /// ////////////// ///
    ///   Conversion   ///
    /// ////////////// ///

    SparseMatSPH mergeNodesRandomWalksMultiThread(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents, bool norm = true, bool weightBySize = true);
    SparseMatSPH mergeNodesRandomWalksSingleThread(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents, bool norm = true, bool weightBySize = true);

    SparseMatSPH mergeNodesRandomWalks(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents, bool norm = true, bool weightBySize = true, bool parallel = true);

    SparseMatSPH mergeNodesDataDistances(const SparseMatSPH& matrixCurrent, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents);
    
    Graph mergeGraphNodes(const GraphInterface& graph, const uint64_t numComponentsMerged, const std::vector<uint64_t>& parents);

    // Convert sparse matrix rows to vector of sparse vectors
    SparseMatSPH matrixToSparseVectors(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& matrix, bool verbose = false);

    void convertGraphToEigenSparse(const GraphBaseInterface& graphView, SparseMatSPH& convertedGraph);

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> createSparseMatrixFromGraph(const GraphBaseInterface& graphView, bool verbose = false);

    // Convert vector of sparse vectors to sparse matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> createSparseMatrixFromVectors(const SparseMatSPH& sparseVectors, bool verbose = false);

    // retains largest k values
    void convertEigenSparseVecToHDILibSparseVec(const SparseVecSPH& eigenVec, int64_t k, SparseVecHDI& hdiVec, bool top = true);

    void convertSparseMatSPHToSparseMatHDI(const SparseMatSPH& matSPH, SparseMatHDI& matHDI);

    /// ///////////// ///
    ///   Distances   ///
    /// ///////////// ///

    // TODO: Rename to computeDistanceMatrices
    // TODO: Remove the componentToPixelMap option

    // Computes A * A^T, uses some extra memory
    // TODO: keep either this one OR createSimilaritiesHDI (the latter basically uses the first + some conversion)
    SparseMatSPH createSimilaritiesEigen(const SparseMatSPH& input, float pruneVal = 0.000f, int verbose = 0, Eigen::Index blockSize = 0, const std::optional<std::reference_wrapper<const vvui64>> componentToPixelMap = std::nullopt);
    
    // Computes A * A^T, uses some extra memory, retains only k largest elements
    SparseMatHDI createSimilaritiesHDI(const SparseMatSPH& input, int64_t k, float pruneVal = 0.000f, int verbose = 0, Eigen::Index blockSize = 0, const std::optional<std::reference_wrapper<const vvui64>> componentToPixelMap = std::nullopt);

    // Computes A * A^T, less memory but slower
    SparseMatSPH createSimilaritiesSPH(const SparseMatSPH& input, float pruneVal = 0.000f, int verbose = 0);

    // Computes A * A^T, 
    SparseMatHDI createSimilarities(const SparseMatSPH& input, int64_t k, float pruneVal = 0.000f, int verbose = 0, const std::optional<std::reference_wrapper<const vvui64>> componentToPixelMap = std::nullopt);

    /// ////////// ///
    ///   General  ///
    /// ////////// ///

    bool isSymmetric(const SparseMatHDI& matSPH);

    bool isSame(const SparseMatHDI& a, const SparseMatHDI& b);

    // keepSingleEntry: does not remove diagonal element if diag is the only value
    uint64_t removeDiagonalElements(SparseMatSPH& sparseMat, bool keepSingleEntry = true);

    void removeElement(SparseVecSPH& sparseMat, size_t elem);

    std::vector<std::pair<uint32_t, float>> findTopK(const SparseVecSPH& vec, size_t k);

    std::vector<std::pair<uint32_t, float>> findBottomK(const SparseVecSPH& vec, size_t k);

} // namespace sph::utils

