#pragma once

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/SparseMatrixAlgorithms.hpp>

#include <catch2/catch_test_macros.hpp>	
#include <Eigen/SparseCore>
#include <hdi/data/map_mem_eff.h>

#include <random>
#include <cstdlib>

inline void fillSparseVectorWithRandomValues(sph::SparseVecSPH& sparseVector, size_t numNonZeros) {
    const size_t size = sparseVector.size();
    sparseVector.reserve(numNonZeros);  // Reserve space for efficiency

    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<Eigen::Index> indexDistribution(0, size - 1);
    std::uniform_real_distribution<float> valueDistribution(0.0f, 1.0f);

    for (size_t i = 0; i < numNonZeros; ++i) {
        Eigen::Index randomIndex = indexDistribution(generator);  // Random index
        float randomValue = valueDistribution(generator);  // Random value

        sparseVector.coeffRef(randomIndex) = randomValue;  // Set value at random index, it's fine if we override 
    }
}

inline sph::SparseMatSPH createRandomSparseMatrix(size_t numRows, size_t numValuesInRow)
{
    sph::SparseMatSPH input(numRows);

    SPH_PARALLEL_ALWAYS
    for (size_t i = 0; i < input.size(); ++i) {
        auto& row = input[i];
        row.resize(numRows);
        fillSparseVectorWithRandomValues(row, numValuesInRow);
        sph::utils::normalizeUnitSparseVector(row);
    }

    return input;
}

inline void checkSameTwoMatricesSPH(const sph::SparseMatSPH& matA, const sph::SparseMatSPH& matB) {
    REQUIRE(matA.size() == matB.size());

    for (size_t entry = 0; entry < matA.size(); entry++)
    {
        const auto& rowA = matA[entry];
        const auto& rowB = matB[entry];

        REQUIRE(rowA.size() == rowB.size());
        REQUIRE(rowA.nonZeros() == rowB.nonZeros());

        for (sph::SparseVecSPH::InnerIterator it(rowA); it; ++it) {

            const auto valA = it.value();
            const auto valB = rowB.coeff(it.index());

            const bool isZeroA = (valA > 0.f || valA < 0.f);
            const bool isZeroB = (valB > 0.f || valB < 0.f);
            REQUIRE(isZeroA == isZeroB);

            const double diff = std::abs(static_cast<double>(it.value()) - static_cast<double>(rowB.coeff(it.index())));
            REQUIRE(diff < 0.00001);
        }
    }
}

inline void checkSameTwoMatricesHDI(const sph::SparseMatHDI& matA, const sph::SparseMatHDI& matB) {
    REQUIRE(matA.size() == matB.size());

    for (size_t rowCounter = 0; rowCounter < matA.size(); rowCounter++)
    {
        auto& row1 = matA[rowCounter].memory();
        auto& row2 = matB[rowCounter].memory();

        REQUIRE(row1.size() == row2.size());
        for (size_t entryCounter = 0; entryCounter < row1.size(); entryCounter++)
        {
            const double diff = std::abs(static_cast<double>(row1[entryCounter].second) - static_cast<double>(row2[entryCounter].second));

            if (row1[entryCounter].first != row2[entryCounter].first || diff >= 0.00001)
            {
                sph::utils::print(row1[entryCounter]);
                sph::utils::print(row2[entryCounter]);

                sph::utils::print(row1);
                sph::utils::print(row2);
            }

            REQUIRE(row1[entryCounter].first == row2[entryCounter].first);
            REQUIRE(diff < 0.00001);
        }
    }
}
