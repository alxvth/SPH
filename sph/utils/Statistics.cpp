#include "Statistics.hpp"

#include "Graph.hpp"
#include "Logger.hpp"

#include <cmath>
#include <limits>

#include <Eigen/SparseCore>
#include <hdi/data/map_mem_eff.h>

namespace sph::utils {

    void printSparseMatrixStats(const SparseMatrixStats& stats, const std::string& sparseMatrixName)
    {
        if(stats.pruneVale >0)
            Log::info("{}: {:.2f}% sparsity, {:.2f}% effective sparsity; {} non-zero entries of {}", sparseMatrixName, stats.sparsity * 100, stats.sparsityEffective * 100, stats.nonZeros, stats.totalEntries);
        else
            Log::info("{}: {:.2f}% sparsity; {} non-zero entries of {}", sparseMatrixName, stats.sparsity * 100, stats.nonZeros, stats.totalEntries);

        Log::info("On average {:.2f} non-zero entries over {} rows", stats.averageNonZeros, std::sqrt(stats.totalEntries));
    }

    SparseMatrixStats sparseMatrixStats(const std::vector<SparseVecSPH>& sparseMatrix, std::optional<float> pruneVal)
    {
        SparseMatrixStats stats;

        if (sparseMatrix.empty())
            return stats;

        const float thresh                  = pruneVal.value_or(std::numeric_limits<float>::min());
        const size_t totalEntries           = sparseMatrix.size() * sparseMatrix.size();
        size_t populatedEntries             = 0;
        size_t populatedEntriesEffective    = 0;

        for (const auto& row : sparseMatrix)
        {
            populatedEntries += row.nonZeros();

            if (thresh > std::numeric_limits<float>::min())
            {
                for (SparseVecSPH::InnerIterator it(row); it; ++it)
                    if (it.value() > thresh)
                        populatedEntriesEffective++;
            }
        }

        stats.totalEntries        = totalEntries;
        stats.nonZeros            = populatedEntries;
        stats.sparsity            = 1.0 - (static_cast<double>(populatedEntries) / totalEntries);
        stats.sparsityEffective   = 1.0 - (static_cast<double>(populatedEntriesEffective) / totalEntries);

        if (pruneVal.has_value())
            stats.averageNonZeros = static_cast<double>(populatedEntriesEffective) / sparseMatrix.size();
        else
            stats.averageNonZeros = static_cast<double>(populatedEntries) / sparseMatrix.size();

        return stats;
    }

    SparseMatrixStats sparseMatrixStats(const std::vector<SparseVecHDI>& sparseMatrix, std::optional<float> pruneVal)
    {
        SparseMatrixStats stats;

        const float thresh = pruneVal.value_or(std::numeric_limits<float>::min());
        const size_t totalEntries = sparseMatrix.size() * sparseMatrix.size();
        size_t populatedEntries = 0;
        size_t populatedEntriesEffective = 0;

        for (const auto& row : sparseMatrix)
        {
            populatedEntries += row.size();

            if (thresh > std::numeric_limits<float>::min())
            {
                for (const auto& entry : row)
                    if (entry.second > thresh)
                        populatedEntriesEffective++;
            }
        }

        stats.totalEntries        = totalEntries;
        stats.nonZeros            = populatedEntries;
        stats.sparsity            = 1.0 - (static_cast<double>(populatedEntries) / totalEntries);
        stats.sparsityEffective   = 1.0 - (static_cast<double>(populatedEntriesEffective) / totalEntries);

        if (pruneVal.has_value())
            stats.averageNonZeros = static_cast<double>(populatedEntriesEffective) / sparseMatrix.size();
        else
            stats.averageNonZeros = static_cast<double>(populatedEntries) / sparseMatrix.size();

        return stats;
    }

    SparseMatrixStats sparseMatrixStats(const GraphInterface* sparseMatrix)
    {
        SparseMatrixStats stats;

        const int64_t numPoints = sparseMatrix->getNumPoints();
        const size_t totalEntries = numPoints * numPoints;
        size_t populatedEntries = 0;
        size_t populatedEntriesEffective = 0;

        for (int64_t i = 0; i < numPoints; i++)
            populatedEntries += sparseMatrix->getK();

        stats.totalEntries      = totalEntries;
        stats.nonZeros          = populatedEntries;
        stats.sparsity          = 1.0 - (static_cast<double>(populatedEntries) / totalEntries);
        stats.sparsityEffective = stats.sparsity;

        stats.averageNonZeros   = static_cast<double>(populatedEntriesEffective) / numPoints;

        return stats;
    }

}
