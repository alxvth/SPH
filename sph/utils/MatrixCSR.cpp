#include "MatrixCSR.hpp"

#include <Eigen/SparseCore>

#include <cassert>
#include <format>
#include <iostream>

namespace sph::utils {

    MatrixCSR convertToCSR(const SparseMatSPH& input) {
        MatrixCSR matrixCSR;
        matrixCSR.row_num = static_cast<int>(input.size());
        matrixCSR.col_num = matrixCSR.row_num;  // we know these are square

        // Compute total non-zeros and max column index
        matrixCSR.nnz = 0;
        for (const auto& row : input) {
            matrixCSR.nnz += row.nonZeros();
        }

        // Allocate memory
        matrixCSR.allocate(matrixCSR.nnz, matrixCSR.row_num);

        // Fill CSR data
        int valueIndex = 0;
        matrixCSR.row_pointers[0] = valueIndex;
        for (int i = 0; i < matrixCSR.row_num; i++) {
            for (SparseVecSPH::InnerIterator it(input[i]); it; ++it) {
                matrixCSR.values[valueIndex] = static_cast<float>(it.value());
                matrixCSR.col_indices[valueIndex] = static_cast<int>(it.index());
                valueIndex++;
            }
            matrixCSR.row_pointers[i + 1] = valueIndex;
        }

        assert(static_cast<uint64_t>(valueIndex) == matrixCSR.nnz);

        return matrixCSR;
    }

    MatrixCSR convertToCSR(const Eigen::SparseMatrix<float, Eigen::RowMajor, int>& mat, int startRow, int startCol, int numRows, int numCols) {
        MatrixCSR matrixCSR;
        matrixCSR.row_num = numRows;
        matrixCSR.col_num = numCols;

        matrixCSR.row_pointers.resize(numRows + 1);

        // Fill CSR data
        int valueIndex = 0;
        matrixCSR.row_pointers[0] = 0;
        for (int i = 0; i < numRows; i++) {
            int originalRow = startRow + i;
            if (originalRow >= mat.rows()) continue;

            // Iterate through non-zeros in this row
            for (Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator it(mat, originalRow); it; ++it) {
                int col = static_cast<int>(it.col() - startCol);
                // Check if this element is within our block
                if (col >= 0 && col < numCols) {
                    matrixCSR.values.push_back(it.value());
                    matrixCSR.col_indices.push_back(col);
                    valueIndex++;
                }
            }
            matrixCSR.row_pointers[i + 1] = valueIndex;
        }

        matrixCSR.nnz = valueIndex;
        matrixCSR.row_pointers_begin = matrixCSR.row_pointers.data();
        matrixCSR.row_pointers_end = matrixCSR.row_pointers.data() + 1;

        return matrixCSR;
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> convertToEigenSparse(const MatrixCSR& matCSR, float pruneValue) {

        std::vector<Eigen::Triplet<float, SparseVecSPH::Index>> triplets;
        triplets.reserve(matCSR.nnz);

        // Iterate through the CSR data and populate the sparse matrix.
        for (int i = 0; i < matCSR.row_num; ++i) {
            for (int j = matCSR.row_pointers[i]; j < matCSR.row_pointers[i + 1]; ++j) {
                if (matCSR.values[j] <= pruneValue)
                    continue;
                triplets.emplace_back(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(matCSR.col_indices[j]), matCSR.values[j]);
            }
        }

        // Create the sparse matrix from the combined triplets
        Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatrix(matCSR.row_num, matCSR.col_num);
        sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
        sparseMatrix.makeCompressed();

        return sparseMatrix;
    }

    Eigen::MatrixXf convertToEigenDense(const MatrixCSR& matCSR) {
        Eigen::MatrixXf denseMatrix = Eigen::MatrixXf::Zero(matCSR.row_num, matCSR.col_num);

        SPH_PARALLEL
        for (int i = 0; i < matCSR.row_num; ++i) {
            for (int j = matCSR.row_pointers[i]; j < matCSR.row_pointers[i + 1]; ++j) {
                denseMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(matCSR.col_indices[j])) = matCSR.values[j];
            }
        }

        return denseMatrix;
    }

    void printCSRMatrix(const MatrixCSR& matrix) {
        for (int i = 0; i < matrix.row_num; ++i) {
            for (int j = 0; j < matrix.col_num; ++j) {
                bool found = false;
                for (int k = matrix.row_pointers[i]; k < matrix.row_pointers[i + 1]; ++k) {
                    if (matrix.col_indices[k] == j) {
                        std::cout << std::format("{:{}.{}f}", matrix.values[k], 4, 3) << " ";
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cout << std::format("{:{}.{}f}", 0.f, 4, 3) << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    void printUpperTriangleCSR(const MatrixCSR& csr) {
        for (int i = 0; i < csr.row_num; ++i) {
            for (int j = csr.row_pointers[i]; j < csr.row_pointers[i + 1]; ++j) {
                int col = csr.col_indices[j];
                if (col >= i) { // Upper triangle condition (including the diagonal)
                    float value = csr.values[j];
                    std::cout << "Row: " << i << ", Col: " << col << ", Value: " << value << std::endl;
                }
            }
        }
    }

    void printLowerTriangleCSR(const MatrixCSR& csr) {
        for (int i = 0; i < csr.row_num; ++i) {
            for (int j = csr.row_pointers[i]; j < csr.row_pointers[i + 1]; ++j) {
                int col = csr.col_indices[j];
                if (col <= i) { // Lower triangle condition (including the diagonal)
                    float value = csr.values[j];
                    std::cout << "Row: " << i << ", Col: " << col << ", Value: " << value << std::endl;
                }
            }
        }
    }

    MatrixCSR transposeCSR(const MatrixCSR& A) {
        MatrixCSR At;
        At.nnz = A.nnz;
        At.row_num = A.col_num;
        At.col_num = A.row_num;
        At.allocate(A.nnz, At.row_num);

        // Count non-zero entries in each column of A to pre-allocate space for At
        std::vector<int> col_counts(A.col_num, 0);
        for (int i = 0; i < A.row_num; ++i) {
            for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j) {
                col_counts[A.col_indices[j]]++;
            }
        }

        // Calculate row pointers for At
        for (int i = 0; i < A.col_num; ++i) {
            At.row_pointers[i + 1] = At.row_pointers[i] + col_counts[i];
        }

        // Populate values and column indices of At
        std::vector<int> current_col_index(A.col_num, 0);  // Tracks the next free slot for each column in At
        for (int i = 0; i < A.row_num; ++i) {
            for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j) {
                const int col = A.col_indices[j];

                const int index = At.row_pointers[col] + current_col_index[col]; // Index in At's arrays
                At.values[index] = A.values[j];
                At.col_indices[index] = i;

                current_col_index[col]++;
            }
        }

        return At;
    }

} // namespace sph::utils
