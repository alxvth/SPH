#pragma once

#include <cstdint>
#include <vector>

#include "CommonDefinitions.hpp"

#include <Eigen/SparseCore>
#include <Eigen/Dense>

namespace sph::utils {

    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/sparse-blas-csr-matrix-storage-format.html
    // zero-based, 3-Array Variation
    // TODO: maybe use uint64_t instead of int
    struct MatrixCSR {
        int row_num = 0;                    // Number of rows
        int col_num = 0;                    // Number of columns
        uint64_t nnz = 0;                   // Number of non-zero values
        std::vector<float> values = {};     // Non-zero values
        std::vector<int> col_indices = {};  // Column indices
        std::vector<int> row_pointers = {}; // Row pointers
        int* row_pointers_begin = nullptr;  // Row pointers begin
        int* row_pointers_end = nullptr;    // Row pointers end

        void allocate(uint64_t nnz, int rows) {
            values.resize(nnz);
            col_indices.resize(nnz);
            row_pointers.resize(rows + 1);
            row_pointers_begin = row_pointers.data();
            row_pointers_end = row_pointers.data() + 1;
        }
    };

    MatrixCSR convertToCSR(const SparseMatSPH& input);
    MatrixCSR convertToCSR(const Eigen::SparseMatrix<float, Eigen::RowMajor, int>& mat, int startRow, int startCol, int numRows, int numCols);
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> convertToEigenSparse(const MatrixCSR& matCSR, float pruneValue = 0);
    Eigen::MatrixXf convertToEigenDense(const MatrixCSR& matCSR);

    void printCSRMatrix(const MatrixCSR& matrix);
    void printUpperTriangleCSR(const MatrixCSR& csr);
    void printLowerTriangleCSR(const MatrixCSR& csr);
    MatrixCSR transposeCSR(const MatrixCSR& A);

} // namespace sph::utils
