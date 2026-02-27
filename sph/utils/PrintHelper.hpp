#pragma once

#include <cstdint>
#include <format>
#include <iostream>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CommonDefinitions.hpp"

namespace Eigen
{
    template <typename Scalar, int Options, typename Index>
    class SparseVector;

    template <typename Scalar, int Options, typename Index>
    class SparseMatrix;

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    class Matrix;

    template <typename Type, int Size>
    using Vec = Matrix<Type, Size, 1, 0, -1, 1>;
}

namespace sph {
    class ImageHierarchy;
}

namespace sph::utils {

    struct GraphBaseInterface;

    namespace stats {
        struct ShortestPathStatistics;
        struct SimilaritiesStatistics;
    }

    /// /////// ///
    /// GENERAL ///
    /// /////// ///

    template<typename T>
    inline void printVector(const std::vector<T>& vec)
    {
        for (const T& val : vec)
            std::cout << val << " ";
        std::cout << std::endl;
    }

    template<typename T>
    inline void printSpan(const std::span<const T>& span)
    {
        for (const T& val : span)
            std::cout << val << " ";
        std::cout << std::endl;
    }

    template<typename T>
    inline void print(const std::span<T>& span)
    {
        printSpan<T>(span);
    }

    template<typename T>
    inline void print(const std::span<const T>& span)
    {
        printSpan<const T>(span);
    }

    template<typename T>
    inline void print(const std::vector<T>& vec)
    {
        printVector<T>(vec);
    }

    template<typename T, typename S, template<typename, typename> class Associative>
    inline void printAssociative(const Associative<T, S>& asc)
    {
        for (auto& [key, value] : asc)
            std::cout << key << " : " << value << "\n";
        std::cout << std::endl;
    }

    template<typename T, typename S>
    inline void print(const std::unordered_map<T, S>& map)
    {
        printAssociative<T, S, std::unordered_map>(map);
    }

    template<typename T, typename S>
    inline void print(const std::pair<T, S>& pair)
    {
        std::cout << pair.first << " : " << pair.second << "\n";
    }

    template<typename T, typename S>
    inline void print(const std::vector<std::pair<T, S>>& pairs)
    {
        for (const auto& pair : pairs)
            print(pair);
        std::cout << std::endl;
    }

    /// //////// ///
    /// SPECIFIC ///
    /// //////// ///

    void printResults(const float* D, const int64_t* I, int64_t k, int64_t nq);
    void printResults(const std::vector<float>& D, const std::vector<int64_t>& I, int64_t k, int64_t nq);

    void printMaps(const std::unordered_map<int64_t, int64_t>& nodes, const std::unordered_map<int64_t, float>& costs);

    void printImageComponents(const ImageHierarchy& ih, uint64_t level);

    struct TransitionMatrixFormatting {
        uint32_t valuePrecision = 3;
        uint32_t valueSpacing = 4;
        uint32_t headerSpacing = 5;
        uint32_t headerLeadSpacing = 1;
    };

    void printSparseVector(const SparseVecSPH& sparseVec, bool stats = false);
    
    template<typename T>
    void printEigenVector(const Eigen::Vec<T, -1>& vec)
    {
        for (std::ptrdiff_t i = 0; i < vec.size(); ++i)
            std::cout << std::format("{:3}", i) << ": " << vec(i) << "\n";

        std::cout << std::endl;
    }

    void printGraphAsList(const GraphBaseInterface& graph);
    void printGraphAsDenseMatrix(const GraphBaseInterface& graph, bool lineNumber = false, TransitionMatrixFormatting tmf = TransitionMatrixFormatting());
    void printGraphNeighbors(std::span<const int64_t> neighborIDs, std::span<const float> distances);

    void printSparseMatrixAsDense(const std::vector<SparseVecSPH>& matrix, bool lineNumber = false, TransitionMatrixFormatting tmf = TransitionMatrixFormatting());
    void printSparseMatrixAsDense(const Eigen::SparseMatrix<float, 0, int>& matrix, bool lineNumber = false, TransitionMatrixFormatting tmf = TransitionMatrixFormatting());
    
    void printSparseMatrixAsDense(const SparseMatHDI& matrix, bool lineNumber = false, TransitionMatrixFormatting tmf = TransitionMatrixFormatting());

    void printShortestPathStatistics(const utils::stats::ShortestPathStatistics& sts);
    void printSimilaritiesStatistics(const utils::stats::SimilaritiesStatistics& sss);

} // namespace sph::utils
