#include "PrintHelper.hpp"

#include "sph/ImageHierarchy.hpp"

#include "Graph.hpp"
#include "Hierarchy.hpp"
#include "Logger.hpp"
#include "ShortestPath.hpp"
#include "Similarities.hpp"

#include <Eigen/SparseCore>
#include <hdi/data/map_mem_eff.h>

#include <cassert>
#include <format>
#include <iostream>

namespace sph::utils {
    template <typename T>
    static inline void _printLine(T&& str)
    {
        std::cout << str << "\n";
    }

    template <typename T>
    static inline void _print(T&& str)
    {
        std::cout << str;
    }

    void printResults(const float* D, const int64_t* I, int64_t k, int64_t nq) {
        int64_t nn = (k > 5) ? 5 : k;
        _printLine("I (5 first results)=");
        for (int64_t i = 0; i < 5; i++) {
            for (int64_t j = 0; j < nn; j++)
                std::cout << std::vformat("{:.5g}", std::make_format_args(I[i * k + j])) << " ";
            _print("\n");
        }

        _printLine("D=");
        for (int64_t i = 0; i < 5; i++) {
            for (int64_t j = 0; j < nn; j++)
                std::cout << std::vformat("{:.7g}", std::make_format_args(D[i * k + j])) << " ";
            _print("\n");
        }

        _printLine("I (5 last results)=");
        for (int64_t i = nq - 5; i < nq; i++) {
            for (int64_t j = 0; j < nn; j++)
                std::cout << std::vformat("{:.5g}", std::make_format_args(I[i * k + j])) << " ";
            _print("\n");
        }

    }

    void printResults(const std::vector<float>& D, const std::vector<int64_t>& I, int64_t k, int64_t nq) {
        printResults(D.data(), I.data(), k, nq);
    }

    void printMaps(const std::unordered_map<int64_t, int64_t>& nodes, const std::unordered_map<int64_t, float>& costs)
    {
        assert(nodes.size() == costs.size());

        for (const auto& [to, from] : nodes)
        {
            std::cout << std::format("{:<8} -> {:<8}: {:.5g}", from, to, costs.at(to)) << "\n";
        }
        std::cout << std::endl;
    }

    void printImageComponents(const ImageHierarchy& ih, uint64_t level)
    {
        const auto rows = ih.getNumRows();
        const auto cols = ih.getNumCols();
        const auto& h   = ih.getHierarchy();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::format("{:2}", h.pixelComponents[level][i * cols + j]) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void printSparseVector(const SparseVecSPH& sparseVec, bool stats)
    {
        float minValue(1), maxValue(0);

        for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
        {
            std::cout << it.index() << ": " << it.value() << ", ";

            if (stats)
            {
                if (it.value() > maxValue) {
                    maxValue = it.value();
                }
                if (it.value() < minValue) {
                    minValue = it.value();
                }
            }
        }
        std::cout << std::endl;

        if (stats)
            std::cout << "Max: " << maxValue << " , Min: " << minValue << ", Values : " << sparseVec.nonZeros() << std::endl;
    }

    void printGraphAsList(const GraphBaseInterface& graph)
    {
        for (int64_t point = 0; point < graph.getNumPoints(); point++)
            for (int64_t neighbor = 0; neighbor < graph.getK(point); neighbor++)
                std::cout << point << " -> " << graph.getNeighborN(point, neighbor) << ": " << graph.getDistanceN(point, neighbor) << '\n';

        std::cout << std::endl;
    }

    void printGraphAsDenseMatrix(const GraphBaseInterface& graph, bool lineNumber, TransitionMatrixFormatting tmf)
    {
        const auto numPoints = graph.getNumPoints();

        if (lineNumber)
        {
            for (uint32_t i = 0; i < tmf.headerLeadSpacing; ++i)
                std::cout << " ";
            for (Eigen::Index i = 0; i < numPoints; ++i)
                std::cout << std::format("{:{}}", i, tmf.headerSpacing) << " ";
            std::cout << "\n";
        }

        for (int64_t outter = 0; outter < numPoints; ++outter)
        {
            if (lineNumber)
                std::cout << std::format("{:2}", outter) << " ";

            for (int64_t inner = 0; inner < numPoints; ++inner)
            {
                float val = 0;

                if (graph.isDirectNeighbor(outter, inner) > -1)
                    val = graph.getDistance(outter, inner);

                std::cout << std::format("{:{}.{}f}", val, tmf.valueSpacing, tmf.valuePrecision) << " ";

            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void printGraphNeighbors(std::span<const int64_t> neighborIDs, std::span<const float> distances)
    {
        if (neighborIDs.size() != distances.size()) {
            Log::error("printGraphNeighbors: inputs have different lengths");
            return;
        }

        // Iterate over the spans using iterators
        auto it1 = neighborIDs.begin();
        auto it2 = distances.begin();
        for (; it1 != neighborIDs.end(); ++it1, ++it2)
            std::cout << *it1 << ": " << *it2 << ", ";

        std::cout << std::endl;
    }

    void printSparseMatrixAsDense(const std::vector<SparseVecSPH>& matrix, bool lineNumber, TransitionMatrixFormatting tmf)
    {
        if (matrix.size() == 0)
            return;

        if (lineNumber)
        {
            for (uint32_t i = 0; i < tmf.headerLeadSpacing; ++i)
                std::cout << " ";
            for (Eigen::Index i = 0; i < matrix[0].size(); ++i)
                std::cout << std::format("{:{}}", i, tmf.headerSpacing) << " ";
            std::cout << "\n";
        }

        for (size_t i = 0; i < matrix.size(); i++)
        {
            const auto& sparseRow = matrix[i];

            if (lineNumber)
                std::cout << std::format("{:2}", i) << " ";

            for (Eigen::Index i = 0; i < sparseRow.size(); ++i) {
                float val = 0;
                if (sparseRow.coeff(i) != 0.0)
                    val = sparseRow.coeff(i);

                std::cout << std::format("{:{}.{}f}", val, tmf.valueSpacing, tmf.valuePrecision) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void printSparseMatrixAsDense(const Eigen::SparseMatrix<float, 0, int>& matrix, bool lineNumber, TransitionMatrixFormatting tmf)
    {
        if (lineNumber)
        {
            for (uint32_t i = 0; i < tmf.headerLeadSpacing; ++i)
                std::cout << " ";
            for (Eigen::Index i = 0; i < matrix.cols(); ++i)
                std::cout << std::format("{:{}}", i, tmf.headerSpacing) << " ";
            std::cout << "\n";
        }

        for (Eigen::Index i = 0; i < matrix.rows(); i++)
        {
            const auto& sparseRow = matrix.row(i);

            if (lineNumber)
                std::cout << std::format("{:2}", i) << " ";

            for (Eigen::Index i = 0; i < sparseRow.size(); ++i) {
                float val = 0;
                if (sparseRow.coeff(i) != 0.0)
                    val = sparseRow.coeff(i);

                std::cout << std::format("{:{}.{}f}", val, tmf.valueSpacing, tmf.valuePrecision) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void printSparseMatrixAsDense(const SparseMatHDI& matrix, bool lineNumber, TransitionMatrixFormatting tmf)
    {
        if (lineNumber)
        {
            for (uint32_t i = 0; i < tmf.headerLeadSpacing; ++i)
                std::cout << " ";
            for (size_t i = 0; i < matrix.size(); ++i)
                std::cout << std::format("{:{}}", i, tmf.headerSpacing) << " ";
            std::cout << "\n";
        }

        for (size_t i = 0; i < matrix.size(); i++)
        {
            auto& sparseRow = matrix[i];
            
            if (lineNumber)
                std::cout << std::format("{:2}", i) << " ";

            for (uint32_t i = 0; i < static_cast<uint32_t>(matrix.size()); ++i) {
                float val = 0;
                if (sparseRow.find(i) != sparseRow.end())
                    val = sparseRow.find(i)->second;

                std::cout << std::format("{:{}.{}f}", val, tmf.valueSpacing, tmf.valuePrecision) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void printShortestPathStatistics(const utils::stats::ShortestPathStatistics& sts)
    {
        Log::info("## ShortestPathStatistics ##");
        Log::info("numCallTotal: {}", sts.numCallTotal);
        Log::info("numCacheLookupSuccess: {}", sts.numCacheLookupSuccess);
        Log::info("numComputeTotal: {}", sts.numComputeTotal);
        Log::info("numComputeAStar: {}", sts.numComputeAStar);
        Log::info("numComputeBoostAStar: {}", sts.numComputeBoostAStar);
        Log::info("numComputeBoostDijkstra: {}", sts.numComputeBoostDijkstra);
    }

    void printSimilaritiesStatistics(const utils::stats::SimilaritiesStatistics& sss)
    {
        Log::info("## SimilaritiesStatistics ##");
        Log::info("numCallTotal: {}", sss.numCallTotal);
        Log::info("numCacheLookupSuccess: {}", sss.numCacheLookupSuccess);
        Log::info("numComputeTotal: {}", sss.numComputeTotal);
    }

} // namespace sph::utils
