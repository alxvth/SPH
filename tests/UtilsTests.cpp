#include "UtilsTests.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "HelperCatch2.hpp"
#include "HelperSPH.hpp"

#include <sph/NearestNeighbors.hpp>
#include <sph/utils/Algorithms.hpp>
#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/Distances.hpp>
#include <sph/utils/EvalIO.hpp>
#include <sph/utils/FileIO.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/GraphBoost.hpp>
#include <sph/utils/GraphUtils.hpp>
#include <sph/utils/HDILibHelper.hpp>
#include <sph/utils/Math.hpp>
#include <sph/utils/MaxSizeDeque.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/TestData.hpp>
#include <sph/utils/Timer.hpp>

#include <catch2/catch_test_macros.hpp>	
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <fmt/base.h>
#include <hdi/data/map_mem_eff.h>

using namespace sph;

namespace sph::utils
{
    // keep this function here for testing
    static Eigen::SparseMatrix<float, Eigen::RowMajor, int> createSparseMatrixFromVectorsSequential(const SparseMatSPH& sparseVectors) 
    {
        if (sparseVectors.empty()) 
            return Eigen::SparseMatrix<float, Eigen::RowMajor, int>();

        const size_t rows = sparseVectors.size();
        Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatrix(rows, rows);
        size_t avgNonZeros = 0;
        for (const auto& vec : sparseVectors)
            avgNonZeros += vec.nonZeros();

        avgNonZeros = (avgNonZeros + rows - 1) / rows;  // Round up

        sparseMatrix.reserve(Eigen::VectorXf::Constant(rows, static_cast<float>(avgNonZeros)));
        for (size_t i = 0; i < rows; ++i) {
            for (SparseVecSPH::InnerIterator it(sparseVectors[i]); it; ++it)
                sparseMatrix.insert(i, it.index()) = it.value();
        }
        sparseMatrix.makeCompressed();
        return sparseMatrix;
    }

}

/* numpy defaults to linear interpolation
>>> import numpy as np
>>> a = np.array([-1, 3, 7, 5, 2, 5, 3, 6, 7, 8, 1, -1, -1, 9])
>>> np.quantile(a, 0)
np.float64(-1.0)
>>> np.quantile(a, 1)
np.float64(9.0)
>>> np.quantile(a, 0.2)
np.float64(0.20000000000000018)
>>> np.quantile(a, 0.35)
np.float64(2.55)
>>> np.quantile(a, 0.4)
np.float64(3.0)
>>> np.quantile(a, 0.5)
np.float64(4.0)
>>> np.quantile(a, 0.6)
np.float64(5.0)
>>> np.quantile(a, 0.8)
np.float64(7.0)
*/
void testComputeQuantile()
{
    std::vector<float> vec = { -1, 3, 7, 5, 2, 5, 3, 6, 7, 8, 1, -1, -1, 9 };

    // -1, -1,  -1,   1,   2,   3,   3,   5,   5,   6,   7,   7,   8,  9
    // -1  -1   -1    1    2    3    3    5    5    6    7    7    8   9  data values, sorted
    //  0   1    2    3    4    5    6    7    8    9   10   11   12  13  numbered
    //  0 7.7 15.4 23.1 30.8 38.5 46.2 53.9 61.5 69.2 76.9 84.6 92.3 100  percent
    //                0    1    2    3    4    5    6    7    8    9  10  numbered, ignored -1
    //                0   10   20   30   40   50   60   70   80   90 100 percent, ignored -1

    float quantile_value            = 0.f;
    float quantile_value_ignore     = 0.f;
    float quantile_value_linear     = 0.f;
    float quantile                  = 0.f;
    std::vector<float> ignore       = { -1.f };

    {
        quantile = 0.0f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == -1.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 1.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE(quantile_value_linear == -1.f);
    }

    {
        quantile = 1.f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 9.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 9.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE(quantile_value_linear == 9.f);
    }

    {
        quantile = 0.2f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 0.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 3.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE_THAT(quantile_value_linear, Catch::Matchers::WithinAbs(0.2f, 0.0001f));
    }

    {
        quantile = 0.35f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 2.5f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 4.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE_THAT(quantile_value_linear, Catch::Matchers::WithinAbs(2.55f, 0.0001f));
    }

    {
        quantile = 0.4f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 3.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 5.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE(quantile_value_linear == 3.f);
    }

    {
        quantile = 0.5f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 4.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 5.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE(quantile_value_linear == 4.f);
    }

    {
        quantile = 0.6f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 5.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 6.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE(quantile_value_linear == 5.f);
    }

    {
        quantile = 0.8f;

        quantile_value = utils::computeQuantile(vec, quantile);
        REQUIRE(quantile_value == 7.f);

        quantile_value_ignore = utils::computeQuantile(vec, quantile, ignore);
        REQUIRE(quantile_value_ignore == 7.f);

        quantile_value_linear = utils::computeQuantile(vec, quantile, {}, 1);
        REQUIRE(quantile_value_linear == 7.f);
    }

}

void testClampValues()
{
    std::vector<float> valuesReference = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };

    {
        std::vector<float> values = valuesReference;
        utils::clampValues(values, 1.f, 8.f);
        REQUIRE(values == vf32{ 1, 1, 2, 3, 4, 5, 6, 7, 8, 8 });
    }

    {
        std::vector<float> values = valuesReference;
        utils::clampValues(values, 3.f, 6.f);
        REQUIRE(values == vf32{ 3, 3, 3, 3, 4, 5, 6, 6, 6, 6 });
    }

    {
        std::vector<float> values = valuesReference;
        utils::clampValues(values, 2.5f, 5.5f);
        REQUIRE(values == vf32{ 2.5, 2.5, 2.5, 3, 4, 5, 5.5, 5.5, 5.5, 5.5 });
    }

}



void testL2()
{
    auto baseline_L2 = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
        assert(a.size() == b.size());
        double sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += static_cast<double>(a[i] - b[i]) * static_cast<double>(a[i] - b[i]);
        return static_cast<float>(std::sqrt(sum));
        };

	std::vector<size_t> vec_size = { 10, 25, 50, 100, 150 };

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (const size_t dims : vec_size)
    {
        std::vector<float> vec1(dims);
        std::vector<float> vec2(dims);

        // Fill vectors with random values
        for (size_t dim = 0; dim < dims; ++dim) {
            vec1[dim] = dist(gen);
            vec2[dim] = dist(gen);
        }

		float baseline = baseline_L2(vec1, vec2);
		float l2 = utils::L2(vec1.data(), vec2.data(), dims);

        REQUIRE_THAT(l2, Catch::Matchers::WithinAbs(baseline, 0.0001f));
    }
}


void testLabelGraphComponents()
{
    {
        // define data and knn
        int64_t k(3), numPoints(24);

        utils::Graph knnGraph;
        knnGraph.numPoints = numPoints;
        knnGraph.knnDistances.resize(numPoints * k, 0);    // not used in this test

        // no node links TO 5, but 5 connects to others
        // {5} and { all others } are the strongly connected components
        knnGraph.knnIndices = {
            0,   1,  2,      1, 8,  9,      2,  4, 20,      3,  7, 15,      4, 16, 18,      5, 16,  0,
            6,   3,  8,      7, 8, 10,      8,  4, 21,      9, 21,  0,     10,  9,  3,     11, 10, 12,
            12, 11, 22,     13, 1,  3,     14,  9, 18,     15,  2, 11,     16,  6, 15,     17, 16, 18,
            18,  1, 19,     19, 2, 23,     20, 23, 21,     21, 23, 15,     22, 17, 14,     23, 13, 12,
        };

        knnGraph.updateFixedNumNeighbors(k);

        assert(knnGraph.isValid());

        auto knnGraphView = knnGraph.getGraphView();
        
        utils::BoostGraph bgraph = utils::createBoostGraph(knnGraphView);

        // results
        std::vector<int64_t> labelsB;
        int64_t numComponentsB = 0;
        int64_t numStrongComponentsB = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents boost graph", "us");
            numComponentsB = utils::labelGraphWeakComponents(bgraph, labelsB);
        }

        REQUIRE(numComponentsB == 1);
        REQUIRE(std::all_of(labelsB.begin(), labelsB.end(), [](auto& in) -> bool { return in == 0; }));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents boost graph", "us");
            numStrongComponentsB = utils::labelGraphStrongComponents(bgraph, labelsB);
        }

        REQUIRE(numStrongComponentsB == 2);

        // results
        std::vector<int64_t> labels;
        int64_t numComponents = 0;
        int64_t numStrongComponents = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents", "us");
            numComponents = utils::labelGraphWeakComponents(knnGraphView, labels);
        }

        REQUIRE(numComponents == 1);
        REQUIRE(std::all_of(labels.begin(), labels.end(), [](auto& in) -> bool { return in == 0; }));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents", "us");
            numStrongComponents = utils::labelGraphStrongComponents(knnGraphView, labels);
        }
        
        REQUIRE(numStrongComponents == 2);

        REQUIRE(labels == labelsB);

    }

    {
        // define data and knn
        int64_t k(3), numPoints(7);

        utils::Graph knnGraph;
        knnGraph.numPoints = numPoints;
        knnGraph.knnDistances.resize(numPoints * k, 0);    // not used in this test

        // {0, 1, 2}, {3, 4, 5}, {6} are strongly connected each
        // 6 links to the other components
        knnGraph.knnIndices = {
            0, 2, 1,
            1, 0, 2,
            2, 1, 0,
            3, 4, 5,
            4, 5, 3,
            5, 3, 4,
            6, 1, 4,
        };

        knnGraph.updateFixedNumNeighbors(k);

        assert(knnGraph.isValid());

        auto knnGraphView = knnGraph.getGraphView();

        utils::BoostGraph bgraph = utils::createBoostGraph(knnGraphView);

        // results
        std::vector<int64_t> labelsB;
        int64_t numComponentsB = 0;
        int64_t numStrongComponentsB = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents boost graph", "us");
            numComponentsB = utils::labelGraphWeakComponents(bgraph, labelsB);
        }

        REQUIRE(numComponentsB == 1);
        REQUIRE(std::all_of(labelsB.begin(), labelsB.end(), [](auto& in) -> bool { return in == 0; }));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents boost graph", "us");
            numStrongComponentsB = utils::labelGraphStrongComponents(bgraph, labelsB);
        }

        REQUIRE(numStrongComponentsB == 3);

        // results
        std::vector<int64_t> labels;
        int64_t numComponents = 0;
        int64_t numStrongComponents = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents", "us");
            numComponents = utils::labelGraphWeakComponents(knnGraphView, labels);
        }

        REQUIRE(numComponents == 1);
        REQUIRE(std::all_of(labels.begin(), labels.end(), [](auto& in) -> bool { return in == 0; }));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents", "us");
            numStrongComponents = utils::labelGraphStrongComponents(knnGraphView, labels);
        }

        REQUIRE(numStrongComponents == 3);

        REQUIRE(labels == labelsB);
    }

    {
        // define data and knn
        int64_t k(3), numPoints(7);

        utils::Graph knnGraph;
        knnGraph.numPoints = numPoints;
        knnGraph.knnDistances.resize(numPoints * k, 0);    // not used in this test

        // {0} {1, 2, 3}, {4, 5, 6} are strongly connected each
        // 0 links to the other components
        knnGraph.knnIndices = {
            0, 1, 4,
            1, 2, 3,
            2, 3, 1,
            3, 1, 2,
            4, 6, 5,
            5, 4, 6,
            6, 5, 4,
        };

        knnGraph.updateFixedNumNeighbors(k);

        assert(knnGraph.isValid());

        auto knnGraphView = knnGraph.getGraphView();

        utils::BoostGraph bgraph = utils::createBoostGraph(knnGraphView);

        // results
        std::vector<int64_t> labelsB;
        int64_t numComponentsB = 0;
        int64_t numStrongComponentsB = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents boost graph", "us");
            numComponentsB = utils::labelGraphWeakComponents(bgraph, labelsB);
        }

        REQUIRE(numComponentsB == 1);
        REQUIRE(std::all_of(labelsB.begin(), labelsB.end(), [](auto& in) -> bool { return in == 0; }));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents boost graph", "us");
            numStrongComponentsB = utils::labelGraphStrongComponents(bgraph, labelsB);
        }

        REQUIRE(numStrongComponentsB == 3);

        // results
        std::vector<int64_t> labels;
        int64_t numComponents = 0;
        int64_t numStrongComponents = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents", "us");
            numComponents = utils::labelGraphWeakComponents(knnGraphView, labels);
        }

        REQUIRE(numComponents == 1);
        REQUIRE(std::all_of(labels.begin(), labels.end(), [](auto& in) -> bool { return in == 0; }));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents", "us");
            numStrongComponents = utils::labelGraphStrongComponents(knnGraphView, labels);
        }

        REQUIRE(numStrongComponents == 3);

        REQUIRE(labels == labelsB);

    }

    {
        // define data and knn
        int64_t k(3), numPoints(9);

        utils::Graph knnGraph;
        knnGraph.numPoints = numPoints;
        knnGraph.knnDistances.resize(numPoints * k, 0);    // not used in this test

        // {0, 1, 2}, {3, 4, 5}, {6, 7, 8} are strongly connected each
        knnGraph.knnIndices = {
            0, 1, 2,
            1, 0, 2,
            2, 1, 0,
            3, 4, 5,
            4, 5, 3,
            5, 3, 4,
            6, 7, 8,
            7, 6, 8,
            8, 7, 6,
        };

        knnGraph.updateFixedNumNeighbors(k);

        assert(knnGraph.isValid());
        
        auto knnGraphView = knnGraph.getGraphView();

        utils::BoostGraph bgraph = utils::createBoostGraph(knnGraphView);

        // results
        std::vector<int64_t> labelsB;
        int64_t numComponentsB = 0;
        int64_t numStrongComponentsB = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents boost graph", "us");
            numComponentsB = utils::labelGraphWeakComponents(bgraph, labelsB);
        }

        REQUIRE(numComponentsB == 3);
        REQUIRE_THAT(labelsB, Catch::Matchers::Equals(std::vector<int64_t>{ 0, 0, 0, 1, 1, 1, 2, 2, 2}));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents boost graph", "us");
            numStrongComponentsB = utils::labelGraphStrongComponents(bgraph, labelsB);
        }

        REQUIRE(numStrongComponentsB == 3);

        // results
        std::vector<int64_t> labels;
        int64_t numComponents = 0;
        int64_t numStrongComponents = 0;

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphWeakComponents", "us");
            numComponents = utils::labelGraphWeakComponents(knnGraphView, labels);
        }

        REQUIRE(numComponents == 3);
        REQUIRE_THAT(labels, Catch::Matchers::Equals(std::vector<int64_t>{ 0, 0, 0, 1, 1, 1, 2, 2, 2}));

        {
            utils::ScopedTimer<std::chrono::microseconds> myTimer("labelGraphStrongComponents", "us");
            numStrongComponents = utils::labelGraphStrongComponents(knnGraphView, labels);
        }

        REQUIRE(numStrongComponents == 3);

        REQUIRE(labels == labelsB);
    }

}

void testSymmetrizeKnnGraph()
{
    // define data and knn
    int64_t k(3), numPoints(24);

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;

    knnGraph.knnIndices = {
        0,   1,  2,      1, 0,  9,      2,  4, 20,      3,  7, 15,      4, 16, 18,      5, 16,  0,
        6,   3,  8,      7, 8, 10,      8,  4, 21,      9, 21,  0,     10,  9,  3,     11, 10, 12,
        12, 11, 22,     13, 1,  3,     14,  9, 18,     15,  2, 11,     16,  6, 15,     17, 16, 18,
        18,  1, 19,     19, 2, 23,     20, 23, 21,     21, 23, 15,     22, 17, 14,     23, 13, 12,
    };

    knnGraph.knnDistances = {
        0,  1,  2,     0, 0.5f,  9,     0,  4, 20,     0,  7, 15,     0, 16, 18,     0,  1,  6,
        0,  3,  8,     0, 8, 10,     0,  4, 21,     0, 21, 25,     0,  3,  9,     0, 10, 12,
        0, 11, 22,     0, 1,  3,     0,  9, 17,     0,  2, 11,     0,  6, 15,     0, 16, 18,
        0,  1, 19,     0, 2, 23,     0, 21, 23,     0, 13, 15,     0, 14, 17,     0, 12, 13,
    };

    knnGraph.updateFixedNumNeighbors(k);

    assert(knnGraph.isValid());

    auto knnGraphView = knnGraph.getGraphView();

    auto symGraph = utils::symmetrizeGraph(knnGraphView);
    auto symGraphOld = utils::symmetrizeGraphOld(knnGraphView);
    auto symGraphEigen = utils::symmetrizeGraphEigen(knnGraphView);

    fmt::println("knnGraphView");
    utils::printGraphAsDenseMatrix(knnGraphView);
    fmt::println("symGraph");
    utils::printGraphAsDenseMatrix(symGraph);
    fmt::println("symGraphOld");
    utils::printGraphAsDenseMatrix(symGraphOld);
    fmt::println("symGraphEigen");
    utils::printGraphAsDenseMatrix(symGraphEigen);

    vi64 n0  = {  0,  1,  2,  5,  9,   };
    vi64 n6  = {  6,  3, 16,  8,       };
    vi64 n12 = { 12, 11, 23, 22        };
    vi64 n12e = { 12, 23, 22, 11,      };   // symmetrizeGraphEigen takes mean not min
    vi64 n18 = { 18,  1, 14,  4, 17, 19};

    REQUIRE_THAT(symGraph.getNeighbors(0), EqualsSpan(std::span<const int64_t>(n0)));
    REQUIRE_THAT(symGraph.getNeighbors(6), EqualsSpan(std::span<const int64_t>(n6)));
    REQUIRE_THAT(symGraph.getNeighbors(12), EqualsSpan(std::span<const int64_t>(n12)));
    REQUIRE_THAT(symGraph.getNeighbors(18), EqualsSpan(std::span<const int64_t>(n18)));

    REQUIRE(symGraph.getNns()[0] == static_cast<int64_t>(n0.size()));
    REQUIRE(symGraph.getNns()[6] == static_cast<int64_t>(n6.size()));
    REQUIRE(symGraph.getNns()[12] == static_cast<int64_t>(n12.size()));
    REQUIRE(symGraph.getNns()[18] == static_cast<int64_t>(n18.size()));

    for (int64_t point = 0; point < symGraph.getNumPoints(); point++)
        for (const auto& neighbor : symGraph.getNeighbors(point))
        {
            REQUIRE(symGraph.isDirectNeighbor(point, neighbor) != -1);
            REQUIRE(symGraph.isDirectNeighbor(neighbor, point) != -1);
        }

    REQUIRE_THAT(symGraphOld.getNeighbors(0), EqualsSpan(std::span<const int64_t>(n0)));
    REQUIRE_THAT(symGraphOld.getNeighbors(6), EqualsSpan(std::span<const int64_t>(n6)));
    REQUIRE_THAT(symGraphOld.getNeighbors(12), EqualsSpan(std::span<const int64_t>(n12)));
    REQUIRE_THAT(symGraphOld.getNeighbors(18), EqualsSpan(std::span<const int64_t>(n18)));

    REQUIRE(symGraphOld.getNns()[0] == static_cast<int64_t>(n0.size()));
    REQUIRE(symGraphOld.getNns()[6] == static_cast<int64_t>(n6.size()));
    REQUIRE(symGraphOld.getNns()[12] == static_cast<int64_t>(n12.size()));
    REQUIRE(symGraphOld.getNns()[18] == static_cast<int64_t>(n18.size()));

    for (int64_t point = 0; point < symGraphOld.getNumPoints(); point++)
        for (const auto& neighbor : symGraphOld.getNeighbors(point))
        {
            REQUIRE(symGraphOld.isDirectNeighbor(point, neighbor) != -1);
            REQUIRE(symGraphOld.isDirectNeighbor(neighbor, point) != -1);
        }

    REQUIRE_THAT(symGraphEigen.getNeighbors(0), EqualsSpan(std::span<const int64_t>(n0)));
    REQUIRE_THAT(symGraphEigen.getNeighbors(6), EqualsSpan(std::span<const int64_t>(n6)));
    REQUIRE_THAT(symGraphEigen.getNeighbors(12), EqualsSpan(std::span<const int64_t>(n12e)));
    REQUIRE_THAT(symGraphEigen.getNeighbors(18), EqualsSpan(std::span<const int64_t>(n18)));

    REQUIRE(symGraphEigen.getNns()[0] == static_cast<int64_t>(n0.size()));
    REQUIRE(symGraphEigen.getNns()[6] == static_cast<int64_t>(n6.size()));
    REQUIRE(symGraphEigen.getNns()[12] == static_cast<int64_t>(n12.size()));
    REQUIRE(symGraphEigen.getNns()[18] == static_cast<int64_t>(n18.size()));

    for (int64_t point = 0; point < symGraphEigen.getNumPoints(); point++)
        for (const auto& neighbor : symGraphEigen.getNeighbors(point))
        {
            REQUIRE(symGraphEigen.isDirectNeighbor(point, neighbor) != -1);
            REQUIRE(symGraphEigen.isDirectNeighbor(neighbor, point) != -1);
        }

}

void testConnectingComponentsKnnGraph()
{
    // create test data that is very sparse
    std::vector<float> positions; 
    std::vector<float> colors;  // unused here
    int64_t numPoints = 1000; 
    int64_t numDims = 3; 
    float noise = 0.2f; 
    uint64_t random_state = 123;

    utils::createSCurve(positions, colors, numPoints, noise, random_state);
    utils::Data data = { std::move(positions), numDims };

    REQUIRE(data.getNumPoints() == numPoints);

    // compute knn and connected components
    auto NN = NearestNeighbors(data);
    NN.setCachingActive(false);

    NearestNeighborsSettings nns;
    nns.numNearestNeighbors = 3;
    nns.knnMetric = utils::KnnMetric::L2;
    nns.knnIndex = utils::KnnIndex::BruteForce;

    NN.compute(nns);

    auto CC = NN.computeConnectedComponents();

    REQUIRE(CC.first > 1);

    vi64 labels;

    // connect and compute connected components again
    auto connectedGraph = NN.connectComponents();

    utils::BoostGraph bgraph = utils::createBoostGraph(connectedGraph);
    auto newNumWCCB = utils::labelGraphWeakComponents(bgraph, labels);
    REQUIRE(newNumWCCB == 1);

    auto newNumWCC = utils::labelGraphWeakComponents(connectedGraph, labels);
    REQUIRE(newNumWCC == newNumWCCB);
}

void testGraphAccess()
{
    // define data and knn
    int64_t k(3), numPointsKnn(24);

    utils::Graph knnGraph;
    knnGraph.numPoints = numPointsKnn;

    knnGraph.knnIndices = {
        0,   1,  2,      1, 8,  9,      2,  4, 20,      3,  7, 15,      4, 16, 18,      5, 16,  0,
        6,   3,  8,      7, 8, 10,      8,  4, 21,      9, 21,  0,     10,  9,  3,     11, 10, 12,
        12, 11, 22,     13, 1,  3,     14,  9, 18,     15,  2, 11,     16,  6, 15,     17, 16, 18,
        18,  1, 19,     19, 2, 23,     20, 23, 21,     21, 23, 15,     22, 17, 14,     23, 13, 12,
    };

    knnGraph.knnDistances = {
        0,  1,  2,     0, 8,  9,     0,  4, 20,     0,  7, 15,     0, 16, 18,     0,  1,  6,
        0,  3,  8,     0, 8, 10,     0,  4, 21,     0, 21, 25,     0,  3,  9,     0, 10, 12,
        0, 11, 22,     0, 1,  3,     0,  9, 17,     0,  2, 11,     0,  6, 15,     0, 16, 18,
        0,  1, 19,     0, 2, 23,     0, 21, 23,     0, 13, 15,     0, 14, 17,     0, 12, 13,
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    const utils::GraphBaseInterface* graphInterface = &knnGraph;

    vi64 n2 = { 2, 4, 20 };
    vi64 n5 = { 5, 16,  0 };
    vi64 n11 = { 11, 10, 12 };

    REQUIRE_THAT(graphInterface->getNeighbors(2), EqualsSpan(std::span<const int64_t>(n2)));
    REQUIRE_THAT(graphInterface->getNeighbors(5), EqualsSpan(std::span<const int64_t>(n5)));
    REQUIRE_THAT(graphInterface->getNeighbors(11), EqualsSpan(std::span<const int64_t>(n11)));

    vf32 d2 = { 0,  4, 20 };
    vf32 d5 = { 0,  1,  6 };
    vf32 d11 = { 0, 10, 12 };

    REQUIRE_THAT(graphInterface->getDistances(2), EqualsSpan(std::span<const float>(d2)));
    REQUIRE_THAT(graphInterface->getDistances(5), EqualsSpan(std::span<const float>(d5)));
    REQUIRE_THAT(graphInterface->getDistances(11), EqualsSpan(std::span<const float>(d11)));

    // define data and nn
    int64_t numPointsNn = 12;

    utils::Graph nnGraph;
    nnGraph.numPoints = numPointsNn;

    nnGraph.knnIndices = {
        0,  1,         1,            2,  4, 20,     3,  7, 15,     4, 16, 18,     5, 16,  0,  2,  6,  7,
        6,  3,  8,     7, 8, 10,     8,  4, 21,     9, 21,  0,    10,  9,  3,    11, 10, 12,
    };

    nnGraph.knnDistances = {
        0,  1,         0,            0,  4, 20,     0,  7, 15,     0, 16, 18,     0,  1,  6,  8,  9, 14,
        0,  3,  8,     0, 8, 10,     0,  4, 21,     0, 21, 25,     0,  3,  9,     0, 10, 12,
    };

    nnGraph.nns = { 2, 1, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3};
    nnGraph.updateOffsets();

    REQUIRE(nnGraph.isValid());

    graphInterface = &nnGraph;

    n2 = { 2, 4, 20 };
    n5 = { 5, 16,  0,  2,  6,  7 };
    n11 = { 11, 10, 12 };

    REQUIRE_THAT(graphInterface->getNeighbors(2), EqualsSpan(std::span<const int64_t>(n2)));
    REQUIRE_THAT(graphInterface->getNeighbors(5), EqualsSpan(std::span<const int64_t>(n5)));
    REQUIRE_THAT(graphInterface->getNeighbors(11), EqualsSpan(std::span<const int64_t>(n11)));

    d2 = { 0,  4, 20 };
    d5 = { 0,  1,  6,  8,  9, 14 };
    d11 = { 0, 10, 12 };

    REQUIRE_THAT(graphInterface->getDistances(2), EqualsSpan(std::span<const float>(d2)));
    REQUIRE_THAT(graphInterface->getDistances(5), EqualsSpan(std::span<const float>(d5)));
    REQUIRE_THAT(graphInterface->getDistances(11), EqualsSpan(std::span<const float>(d11)));

}

void testMath()
{
    vf32 v1 = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

    auto m11 = utils::computeMean(v1, 6);   // one point, six dimensions
    REQUIRE(m11 == vf32{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });

    auto m12 = utils::computeMean(v1, 3);   // two points, three dimensions
    REQUIRE(m12 == vf32{ 2.5f, 3.5f, 4.5f });

    auto m13 = utils::computeMean(v1, 2);   // three points, two dimensions
    REQUIRE(m13 == vf32{ 3.f, 4.f });

    auto m14 = utils::computeMean(v1, 1);   // six points, one dimension
    REQUIRE(m14 == vf32{ 3.5f });

    vf32 v2 = { 1.4f, 1.4f, 33.3f, -4.0f, 5.0f, 6.0f };

    auto m21 = utils::computeMean(v2, 6);
    REQUIRE(m21 == vf32{ 1.4f, 1.4f, 33.3f, -4.f, 5.f, 6.f });

    auto m22 = utils::computeMean(v2, 3);
    REQUIRE_THAT(m22, Catch::Matchers::Approx(vf32{ -1.3f, 3.2f, 19.65f }).margin(0.01f));

    auto m23 = utils::computeMean(v2, 2);
    REQUIRE_THAT(m23, Catch::Matchers::Approx(vf32{ 13.23333333f, 1.13333333f }).margin(0.01f));

    auto m24 = utils::computeMean(v2, 1);
    REQUIRE_THAT(m24, Catch::Matchers::Approx(vf32{ 7.18333333f }).margin(0.01f));

}

void testEigenSparseMatrix()
{
    sph::SparseMatSPH input;

    size_t numRows = 5;
    input.resize(numRows);
    for (auto& row : input)
        row.resize(numRows);

    // Each row sums to 1
    input[0].insert(0) = 0.2f;
    input[0].insert(2) = 0.4f;
    input[0].insert(4) = 0.4f;
    input[1].insert(1) = 0.3f;
    input[1].insert(2) = 0.7f;
    input[2].insert(2) = 0.5f;
    input[2].insert(4) = 0.5f;
    input[3].insert(2) = 0.8f;
    input[3].insert(3) = 0.1f;
    input[3].insert(4) = 0.1f;
    input[4].insert(0) = 0.6f;
    input[4].insert(2) = 0.4f;

    for(const auto& row : input)
        REQUIRE(row.sum() == 1.0f);

    fmt::println("input");
    utils::printSparseMatrixAsDense(input, true);

    // MANUAL
    // auto dist = [&input](size_t id1, size_t id2) -> float {
    //     // assumes probDist1 and probDist2 are normalized
    //     const auto& probDist1 = input[id1];
    //     const auto& probDist2 = input[id2];

    //     float bc = 0.f;

    //     for (SparseVecSPH::InnerIterator it1(probDist1); it1; ++it1) {
    //         const float value1 = it1.value();
    //         assert(value1 > 0.0f);
    //         const float value2 = probDist2.coeff(it1.index()); // get the corresponding value from normDist2, 0 if not present
    //         if (value2 > 0.0f) {
    //             bc += std::sqrt(value1 * value2);
    //         }
    //     }
    //     return bc;
    //     };

    SparseMatSPH sph_sqrt;
    std::copy(input.begin(), input.end(), std::back_inserter(sph_sqrt));

    SPH_PARALLEL
    for (size_t i = 0; i < input.size(); ++i) {
        float* values = sph_sqrt[i].valuePtr();
        Eigen::Index nonZeros = sph_sqrt[i].nonZeros();
        for (Eigen::Index j = 0; j < nonZeros; ++j) {
            values[j] = std::sqrt(values[j]);
        }
    }

    sph::SparseMatSPH sph_output;
    sph_output.resize(input.size());

    SPH_PARALLEL
    for (size_t i = 0; i < sph_sqrt.size(); ++i) {
        sph_output[i].resize(input.size());
        for (size_t j = 0; j < sph_sqrt.size(); ++j) {
            sph_output[i].insert(j) = sph_sqrt[i].dot(sph_sqrt[j]);
        }
    }

    sph::SparseMatSPH sph_intermediate = sph_output;

    SPH_PARALLEL
    for (size_t i = 0; i < sph_output.size(); ++i) {
        for (SparseVecSPH::InnerIterator it(sph_output[i]); it; ++it) {
            it.valueRef() = 1.f - it.value();
        }
    }

    // EIGEN

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> intermediate  = utils::createSparseMatrixFromVectors(input);

    {
        Eigen::SparseMatrix<float, Eigen::RowMajor, int> intermediate2 = utils::createSparseMatrixFromVectorsSequential(input);   // test parallel
        REQUIRE(intermediate.isApprox(intermediate2, 0.0001f));
    }

    // Compute sqrt of each element
    {
        float* values = intermediate.valuePtr();
        Eigen::Index nonZeros = intermediate.nonZeros();

        SPH_PARALLEL
        for (Eigen::Index i = 0; i < nonZeros; ++i) {
            values[i] = std::sqrt(values[i]);
        }
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> inverted = (intermediate * Eigen::SparseMatrix<float, 0, int>(intermediate.transpose())).pruned();

    Eigen::SparseMatrix<float, Eigen::RowMajor, int> multiplied = inverted;

    // invert each element
    {
        float* values = inverted.valuePtr();
        Eigen::Index nonZeros = inverted.nonZeros();

        SPH_PARALLEL
            for (Eigen::Index i = 0; i < nonZeros; ++i) {
                values[i] = 1.f - values[i];
            }
    }

    sph::SparseMatSPH output = utils::matrixToSparseVectors(inverted);

    // PRINT

    //fmt::println("intermediate");
    //utils::printSparseMatrixAsDense(intermediate, true);

    fmt::println("1. intermediate (sqrt)");
    utils::printSparseMatrixAsDense(intermediate, true);

    fmt::println("1. sph_sqrt");
    utils::printSparseMatrixAsDense(sph_sqrt, true);

    //fmt::println("transposed");
    //utils::printSparseMatrixAsDense(Eigen::SparseMatrix<float, 0, int>(intermediate.transpose()), true);

    fmt::println("2. multiplied");
    utils::printSparseMatrixAsDense(multiplied, true);

    //fmt::println("multiplied (1-x)");
    //utils::printSparseMatrixAsDense(multiplied, true);

    fmt::println("2. sph_intermediate");
    utils::printSparseMatrixAsDense(sph_intermediate, true);

    fmt::println("3. output");
    utils::printSparseMatrixAsDense(output, true);

    fmt::println("3. sph_output");
    utils::printSparseMatrixAsDense(sph_output, true);


    // SHORT

    auto res = utils::createSimilaritiesEigen(input, 0.f);
    auto res_par = utils::createSimilaritiesEigen(input, 0.f, 0, 2);
    auto sph_res = utils::createSimilaritiesSPH(input, 0.f);

    fmt::println("4. res");
    utils::printSparseMatrixAsDense(res, true);

    fmt::println("4. res_par");
    utils::printSparseMatrixAsDense(res_par, true);

    fmt::println("4. sph_res");
    utils::printSparseMatrixAsDense(sph_res, true);

    REQUIRE(res.size() == sph_res.size());
    REQUIRE(res_par.size() == sph_res.size());

    for(size_t i = 0; i < res.size(); i++)
        REQUIRE(res[i].isApprox(sph_res[i], 0.0001f));

    for(size_t i = 0; i < res.size(); i++)
        REQUIRE(res[i].isApprox(res_par[i], 0.0001f));

    checkSameTwoMatricesSPH(res, res_par);
    checkSameTwoMatricesSPH(res, sph_res);

    // Conversion to HDI Matrix

    int64_t k = 2;

    SparseMatHDI hdiMat(numRows);

    for (int64_t i = 0; i < static_cast<int64_t>(numRows); ++i)
        utils::convertEigenSparseVecToHDILibSparseVec(res[i], k, hdiMat[i], /*top=*/ false);

    SparseMatHDI hdiMat2 = utils::createSimilaritiesHDI(input, k, 0.f, false, 2);
    fmt::println("5. hdiMat");
    utils::printSparseMatrixAsDense(hdiMat);

    fmt::println("5. hdiMat2");
    utils::printSparseMatrixAsDense(hdiMat2);

    REQUIRE(hdiMat.size() == hdiMat2.size());

    for (size_t i = 0; i < hdiMat.size(); i++)
    {
        auto& a = hdiMat[i].memory();
        auto& b = hdiMat2[i].memory();

        REQUIRE(a.size() == b.size());

        for (size_t j = 0; j < a.size(); j++)
        {
            REQUIRE(a[j].first == b[j].first);
            REQUIRE(a[j].second == b[j].second);
        }

    }

}

void testMaxSizeDeque()
{
    size_t k = 3;

    utils::SortedMaxPairSizeDeque sd(k);
    utils::SortedMaxPairSizeDequeR sdr(k);

    sd.insert(0.8f, 5);
    sd.insert(0.4f, 3);
    sd.insert(0.3f, 0);
    sd.insert(0.9f, 1);

    sdr.insert(5, 0.8f);
    sdr.insert(3, 0.4f);
    sdr.insert(0, 0.3f);
    sdr.insert(1, 0.9f);

    fmt::println("SortedMaxPairSizeDeque");
    utils::printAssociative(sd.getData());

    fmt::println("SortedMaxPairSizeDequeR");
    utils::printAssociative(sdr.getData());

    SparseVecSPH vec;
    vec.resize(10);
    vec.insert(5) = 0.8f;
    vec.insert(3) = 0.4f;
    vec.insert(0) = 0.3f;
    vec.insert(1) = 0.9f;
    vec.insert(2) = 0.2f;
    vec.insert(4) = 0.1f;
    vec.insert(6) = 0.0f;

    fmt::println("findTopK");
    auto topK = utils::findTopK(vec, k);
    utils::print(topK);

    REQUIRE(topK.size() == k);
    REQUIRE(topK.at(0) == std::pair<uint32_t, float>{1, 0.9f});
    REQUIRE(topK.at(1) == std::pair<uint32_t, float>{3, 0.4f});
    REQUIRE(topK.at(2) == std::pair<uint32_t, float>{5, 0.8f});

}

void testNodeMerging()
{
    const size_t numRowsTest = 100000;
    const size_t numValuesInRowTest = 10000;
    const uint64_t numComponentsMerged = 2000;
    const bool norm = false;
    const bool weightBySize = true;

    // create parents
    std::vector<uint64_t> parents(numRowsTest);
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<Eigen::Index> indexDistribution(0, numComponentsMerged - 1);
    for (auto& num : parents)
        num = indexDistribution(generator);

    const sph::SparseMatSPH matrixCurrent = createRandomSparseMatrix(numRowsTest, numValuesInRowTest);

    auto startDoublePass = utils::now();
    auto mergedDoublePass = utils::mergeNodesRandomWalks(matrixCurrent, numComponentsMerged, parents, norm, weightBySize, /*parallel*/ true );
    fmt::println("mergedDoublePass: {} ms", utils::timeSince(startDoublePass));

    auto startSinglePass = utils::now();
    auto mergedSinglePass = utils::mergeNodesRandomWalks(matrixCurrent, numComponentsMerged, parents, norm, weightBySize, /*parallel*/ false);
    fmt::println("mergedSinglePass: {} ms", utils::timeSince(startSinglePass));

    checkSameTwoMatricesSPH(mergedSinglePass, mergedDoublePass);

}

/* Python test code
import numpy as np

size = 5  # Number of columns

matrix = np.random.rand(size, size)
np.fill_diagonal(matrix, 0)
matrix[0, 3] = 0
matrix[2, 0] = 0

print("Matrix:\n", matrix)

symmetrized_matrix = matrix + matrix.T - matrix * matrix.T
print("Symmetrized Matrix:\n", symmetrized_matrix)

for i in range(size):
    for j in range(i + 1, size):  # Loop through upper triangle
        # Apply the symmetrization formula
        matrix[i, j] = matrix[i, j] + matrix[j, i] - matrix[i, j] * matrix[j, i]
        matrix[j, i] = matrix[i, j]

print("Symmetrized Matrix:\n", matrix)
*/
void testSymmetrizeDistUMAP()
{
    using pair_vec = hdi::data::MapMemEff<uint32_t, float>::storage_type;

    size_t size = 5;
    SparseMatHDI probDist(size);
    SparseMatHDI probDist2(size);

    //  Matrix:
    //  [[0.         0.14390016 0.06372394 0.         0.07803991]
    //  [0.74610553 0.         0.13938483 0.22019774 0.61344256]
    //  [0.         0.73846202 0.         0.61216056 0.82782903]
    //  [0.47865895 0.65175803 0.56427383 0.         0.32879395]
    //  [0.68851214 0.91365522 0.61616717 0.11410742 0.]]

    pair_vec vals_0 = { {1, 0.14390016f}, {2, 0.06372394f}, {4, 0.07803991f} };
    probDist[0].initialize(vals_0.cbegin(), vals_0.cend());
    probDist2[0].initialize(vals_0.cbegin(), vals_0.cend());

    pair_vec vals_1 = { {0, 0.74610553f}, {2, 0.13938483f}, {3, 0.22019774f}, {4, 0.61344256f} };
    probDist[1].initialize(vals_1.cbegin(), vals_1.cend());
    probDist2[1].initialize(vals_1.cbegin(), vals_1.cend());

    pair_vec vals_2 = { {1, 0.73846202f}, {3, 0.61216056f}, {4, 0.82782903f} };
    probDist[2].initialize(vals_2.cbegin(), vals_2.cend());
    probDist2[2].initialize(vals_2.cbegin(), vals_2.cend());

    pair_vec vals_3 = { {0, 0.47865895f}, {1, 0.65175803f}, {2, 0.56427383f}, {4, 0.32879395f} };
    probDist[3].initialize(vals_3.cbegin(), vals_3.cend());
    probDist2[3].initialize(vals_3.cbegin(), vals_3.cend());

    pair_vec vals_4 = { {0, 0.68851214f}, {1, 0.91365522f}, {2, 0.61616717f}, {3, 0.11410742f} };
    probDist[4].initialize(vals_4.cbegin(), vals_4.cend());
    probDist2[4].initialize(vals_4.cbegin(), vals_4.cend());

    fmt::println("Orig matrix:");
    utils::printSparseMatrixAsDense(probDist);

    CHECK(utils::isSame(probDist, probDist2));
    CHECK(!utils::isSymmetric(probDist));

    utils::symmetrizeUMAP(probDist);

    //  Symmetrized Matrix (UMAP):
    //  [[0.        0.78264098 0.06372394 0.47865895 0.71282063]
    //  [0.78264098 0.         0.77491645 0.72844012 0.96662278]
    //  [0.06372394 0.77491645 0.         0.83100821 0.93391513]
    //  [0.47865895 0.72844012 0.83100821 0.         0.40538354]
    //  [0.71282063 0.96662278 0.93391513 0.40538354 0.]]

    fmt::println("Symmetric matrix (UMAP):");
    utils::printSparseMatrixAsDense(probDist);

    REQUIRE(utils::isBasicallyEqual(probDist[0][1], 0.78264098));
    REQUIRE(utils::isBasicallyEqual(probDist[0][2], 0.06372394));
    REQUIRE(utils::isBasicallyEqual(probDist[0][3], 0.47865895));
    REQUIRE(utils::isBasicallyEqual(probDist[0][4], 0.71282063));

    REQUIRE(utils::isBasicallyEqual(probDist[1][2], 0.77491645));
    REQUIRE(utils::isBasicallyEqual(probDist[1][3], 0.72844012));
    REQUIRE(utils::isBasicallyEqual(probDist[1][4], 0.96662278));

    REQUIRE(utils::isBasicallyEqual(probDist[2][3], 0.83100821));
    REQUIRE(utils::isBasicallyEqual(probDist[2][4], 0.93391513));

    REQUIRE(utils::isBasicallyEqual(probDist[3][4], 0.40538354));

    REQUIRE(utils::isSymmetric(probDist));

    //  Symmetrized Matrix (t-SNE):
    //  [[0.        0.44500284 0.03186197 0.23932948 0.38327603]
    //  [0.44500284 0.         0.43892343 0.43597789 0.76354889]
    //  [0.03186197 0.43892343 0.         0.58821719 0.7219981]
    //  [0.23932948 0.43597789 0.58821719 0.         0.22145069]
    //  [0.38327603 0.76354889 0.7219981  0.22145069 0.]]

    fmt::println("Symmetric matrix (t-SNE):");
    utils::symmetrizeTSNE(probDist2);

    utils::printSparseMatrixAsDense(probDist2);

    REQUIRE(utils::isBasicallyEqual(probDist2[0][1], 0.44500284));
    REQUIRE(utils::isBasicallyEqual(probDist2[0][2], 0.03186197));
    REQUIRE(utils::isBasicallyEqual(probDist2[0][3], 0.23932948));
    REQUIRE(utils::isBasicallyEqual(probDist2[0][4], 0.38327603));

    REQUIRE(utils::isBasicallyEqual(probDist2[1][2], 0.43892343));
    REQUIRE(utils::isBasicallyEqual(probDist2[1][3], 0.43597789));
    REQUIRE(utils::isBasicallyEqual(probDist2[1][4], 0.76354889));

    REQUIRE(utils::isBasicallyEqual(probDist2[2][3], 0.58821719));
    REQUIRE(utils::isBasicallyEqual(probDist2[2][4], 0.7219981));

    REQUIRE(utils::isBasicallyEqual(probDist2[3][4], 0.22145069));

    REQUIRE(utils::isSymmetric(probDist2));

}

void testIOCompressedSparseMatHDIBinary()
{
    // Pairs of numPoints, numEntries
    std::vector<std::pair<size_t, size_t>> settings = { {1'000'000, 1'000}, {100'000, 1'000}, {500'000, 500} };

    auto createRandomSparseMatHDI = [](size_t numPoints, size_t numEntries) -> SparseMatHDI {
        SparseMatHDI matrix;
        matrix.resize(numPoints);

        SPH_PARALLEL_ALWAYS
            for (size_t p = 0; p < numPoints; p++)
            {
                auto& mem = matrix[p].memory();

                std::random_device rd; // Seed
                std::mt19937 gen(rd()); // Mersenne Twister engine
                std::uniform_int_distribution<uint32_t> int_dist(0u, static_cast<uint32_t>(numPoints - 1)); // Range for int values
                std::uniform_real_distribution<float> float_dist(0.0f, 1.0f); // Range for float values

                for (size_t e = 0; e < numEntries; e++)
                    mem.emplace_back(int_dist(gen), float_dist(gen));

                std::sort(mem.begin(), mem.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                    });
            }
        return matrix;
        };

    for (const auto& setting : settings)
    {
        const size_t numPoints = setting.first;
        const size_t numEntries = setting.second;

        fmt::println("testIOCompressedSparseMatHDIBinary: numPoints {0}, numEntries {1}", numPoints, numEntries);

        // Create dummy data
        const SparseMatHDI matrix = createRandomSparseMatHDI(numPoints, numEntries);

        // save to disk
        const std::filesystem::path saveDir = std::filesystem::temp_directory_path() / "SPH_TEST";
        const std::filesystem::path savePath = saveDir / "SparseMatHDI.bin";

        fmt::println("testIOCompressedSparseMatHDIBinary: Temp folder {}", saveDir.string());

        if (!utils::ensurePathExists(saveDir))
        {
            fmt::println("testIOCompressedSparseMatHDIBinary: cannot create Temp folder, doing nothing");
            return;
        }

        const bool writeSuccess = utils::writeCompressedSparseMatHDIToBinary(savePath.string(), matrix);

        REQUIRE(writeSuccess);

        // load from disk
        SparseMatHDI matrixLoaded;
        const bool loadSuccess = utils::loadCompressedSparseMatHDIFromBinary(savePath.string(), matrixLoaded);

        REQUIRE(loadSuccess);

        fmt::println("testIOCompressedSparseMatHDIBinary: Checking...");

        // check same
        REQUIRE(matrixLoaded.size() == matrix.size());

        //SPH_PARALLEL
        for (size_t p = 0; p < numPoints; p++)
        {
            auto& mem = matrix[p].memory();
            auto& memLoaded = matrixLoaded[p].memory();

            REQUIRE(mem.size() == memLoaded.size());

        //    SPH_PARALLEL
            for (size_t e = 0; e < mem.size(); e++)
            {
                REQUIRE(mem[e].first == memLoaded[e].first);
                CHECK_THAT(mem[e].second, Catch::Matchers::WithinRel(memLoaded[e].second, 0.0001f));
            }
        }

        // Clean up
        if (!std::filesystem::remove(savePath))
            fmt::println("testIOCompressedSparseMatHDIBinary: cannot delete temp file {}", savePath.string());

        if (!std::filesystem::remove(saveDir))
            fmt::println("testIOCompressedSparseMatHDIBinary: cannot delete temp folder {}", saveDir.string());

    } // for

}


void testIOCompressedSparseMatSPHBinary()
{
    // Pairs of numPoints, numEntries
    std::vector<std::pair<size_t, size_t>> settings = { {1'000'000, 1'000}, {100'000, 1'000}, {500'000, 500} };

    for (const auto& setting : settings)
    {
        const size_t numPoints = setting.first;
        const size_t numEntries = setting.second;

        fmt::println("testIOCompressedSparseMatSPHBinary: numPoints {0}, numEntries {1}", numPoints, numEntries);

        // Create dummy data
        const SparseMatSPH matrix = createRandomSparseMatrix(numPoints, numEntries);

        // save to disk
        const std::filesystem::path saveDir = std::filesystem::temp_directory_path() / "SPH_TEST";
        const std::filesystem::path savePath = saveDir / "SparseMatSPH.bin";

        fmt::println("testIOCompressedSparseMatSPHBinary: Temp folder {}", saveDir.string());

        if (!utils::ensurePathExists(saveDir))
        {
            fmt::println("testIOCompressedSparseMatSPHBinary: cannot create Temp folder, doing nothing");
            return;
        }

        const bool writeSuccess = utils::writeCompressedSparseMatSPHToBinary(savePath.string(), matrix);

        REQUIRE(writeSuccess);

        // load from disk
        SparseMatSPH matrixLoaded;
        const bool loadSuccess = utils::loadCompressedSparseMatSPHFromBinary(savePath.string(), matrixLoaded);

        REQUIRE(loadSuccess);

        fmt::println("testIOCompressedSparseMatSPHBinary: Checking...");

        checkSameTwoMatricesSPH(matrix, matrixLoaded);

        // Clean up
        if (!std::filesystem::remove(savePath))
            fmt::println("testIOCompressedSparseMatSPHBinary: cannot delete temp file {}", savePath.string());

        if (!std::filesystem::remove(saveDir))
            fmt::println("testIOCompressedSparseMatSPHBinary: cannot delete temp folder {}", saveDir.string());

    } // for

}

void testIOCompressedVecBinary()
{
    // Pairs of numPoints, numEntries
    std::vector<size_t> settings = { 1'000'000'000, 1'000'000 };

    for (const size_t& numPoints : settings)
    {
        fmt::println("testIOCompressedVecBinary: numPoints {0}", numPoints);

        // Create dummy data
        std::vector<float> vec(numPoints);

        std::random_device rd;  // Seed
        std::mt19937 gen(rd()); // Mersenne Twister RNG
        std::uniform_real_distribution<float> dist(0.f, 100'000.f);

        for (float& num : vec) {
            num = dist(gen);
        }

        // save to disk
        const std::filesystem::path saveDir = std::filesystem::temp_directory_path() / "SPH_TEST";
        const std::filesystem::path savePath = saveDir / "vec.bin";

        fmt::println("testIOCompressedVecBinary: Temp folder {}", saveDir.string());

        if (!utils::ensurePathExists(saveDir))
        {
            fmt::println("testIOCompressedVecBinary: cannot create Temp folder, doing nothing");
            return;
        }

        const bool writeSuccess = utils::writeCompressedVecToBinary(savePath.string(), vec);

        REQUIRE(writeSuccess);

        // load from disk
        std::vector<float> vecLoaded;
        const bool loadSuccess = utils::loadCompressedVecFromBinary(savePath.string(), vecLoaded);

        REQUIRE(loadSuccess);

        fmt::println("testIOCompressedVecBinary: Checking...");

        REQUIRE(vecLoaded.size() == numPoints);
        REQUIRE(vecLoaded.size() == vec.size());

        for (size_t i = 0; i < vec.size(); i++) {
            double diff = std::abs(static_cast<double>(vecLoaded[i]) - static_cast<double>(vecLoaded[i]));
            REQUIRE(diff < 0.00001);
        }

        // Clean up
        if (!std::filesystem::remove(savePath))
            fmt::println("testIOCompressedVecBinary: cannot delete temp file {}", savePath.string());

        if (!std::filesystem::remove(saveDir))
            fmt::println("testIOCompressedVecBinary: cannot delete temp folder {}", saveDir.string());

    } // for

}
