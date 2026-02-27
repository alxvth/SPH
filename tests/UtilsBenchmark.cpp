#include "UtilsBenchmark.hpp"

#include "HelperSPH.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include <sph/NearestNeighbors.hpp>
#include <sph/utils/AStar.hpp>
#include <sph/utils/AStarBoost.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/Distances.hpp>
#include <sph/utils/GraphUtils.hpp>
#include <sph/utils/Math.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/SparseMatrixAlgorithms.hpp>
#include <sph/utils/TestData.hpp>
#include <sph/utils/Timer.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>	
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <fmt/base.h>

using namespace sph;

constexpr float eps = 0.0001f;

static float distanceL2(const std::vector<float>& data, int64_t numDimensions, int64_t startID, int64_t endID)
{
    return utils::L2(data.data() + startID * numDimensions, data.data() + endID * numDimensions, numDimensions);
}

static Eigen::MatrixXf computePairwiseDistanceMatrix(const std::vector<std::vector<float>>& u, const std::vector<std::vector<float>>& v) {
    size_t m = u.size();
    size_t n = v.size();

    Eigen::MatrixXf distanceMatrix(m, n);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float dx = u[i][0] - v[j][0];
            float dy = u[i][1] - v[j][1];
            distanceMatrix(i, j) = std::sqrt(dx * dx + dy * dy);
        }
    }

    return distanceMatrix;
}

static float directedHausdorffDistance(const Eigen::MatrixXf& distanceMatrix) {
    float maxMinDistance = -std::numeric_limits<float>::infinity();

    for (Eigen::Index i = 0; i < distanceMatrix.rows(); ++i) {
        float minDistance = std::numeric_limits<float>::infinity();

        for (Eigen::Index j = 0; j < distanceMatrix.cols(); ++j)
            minDistance = std::min(minDistance, distanceMatrix(i, j));

        maxMinDistance = std::max(maxMinDistance, minDistance);
    }

    return maxMinDistance;
}

static float symmetricHausdorffDistanceLoops(const Eigen::MatrixXf& distanceMatrix) {
    float forwardDistance = directedHausdorffDistance(distanceMatrix);
    float reverseDistance = directedHausdorffDistance(distanceMatrix.transpose());

    return std::max(forwardDistance, reverseDistance);
}

static float computeHausdorffDistanceSparse(const Eigen::SparseMatrix<float>& distanceMatrix) {
    float maxMinDistance = -std::numeric_limits<float>::infinity();

    for (Eigen::Index k = 0; k < distanceMatrix.outerSize(); ++k) {
        float minDistance = std::numeric_limits<float>::infinity();

        for (Eigen::SparseMatrix<float>::InnerIterator it(distanceMatrix, k); it; ++it) {
            minDistance = std::min(minDistance, it.value());
        }

        maxMinDistance = std::max(maxMinDistance, minDistance);
    }

    return maxMinDistance;
}

static float symmetricHausdorffDistanceSparse(const Eigen::SparseMatrix<float>& distanceMatrix) {
    float forwardDistance = computeHausdorffDistanceSparse(distanceMatrix);
    float reverseDistance = computeHausdorffDistanceSparse(distanceMatrix.transpose());

    return std::max(forwardDistance, reverseDistance);
}

void bm_hausdorff()
{
/*
// Python comparison:
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
from scipy.spatial.distance import directed_hausdorff
import numpy as np

u = np.array([(2.6, 5.1),
              (3.4, 1.9),
              (6.4, 0.8),
              (3.7, 8.0)])

v = np.array([(0.4, 2.1),
              (6.3, 9.1),
              (4.6, 8.0),
              (4.2, 0.8)])

dist = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
print(dist)
>>> 3.5227829907617076
*/
    std::vector<std::vector<float>> u{ {2.6f, 5.1f}, {3.4f, 1.9f}, {6.4f, 0.8f}, {3.7f, 8.0f} };
    std::vector<std::vector<float>> v{ {0.4f, 2.1f}, {6.3f, 9.1f}, {4.6f, 8.0f}, {4.2f, 0.8f} };

    Eigen::MatrixXf distanceMatrix = computePairwiseDistanceMatrix(u, v);

    Eigen::SparseMatrix<float> sparseDistanceMatrix;
    sparseDistanceMatrix.resize(distanceMatrix.rows(), distanceMatrix.cols());

    for (Eigen::Index row = 0; row < distanceMatrix.rows(); ++row)
        for (Eigen::Index col = 0; col < distanceMatrix.cols(); ++col)
            sparseDistanceMatrix.insert(row,  col) = distanceMatrix(row, col);


    CHECK_THAT(symmetricHausdorffDistanceLoops(distanceMatrix), Catch::Matchers::WithinRel(3.52278f, eps));
    CHECK_THAT(utils::symmetricHausdorffDistance(distanceMatrix), Catch::Matchers::WithinRel(3.52278f, eps));
    CHECK_THAT(symmetricHausdorffDistanceSparse(sparseDistanceMatrix), Catch::Matchers::WithinRel(3.52278f, eps));

    BENCHMARK("Hausdorff Loop") {
        return symmetricHausdorffDistanceLoops(distanceMatrix);
    };

    BENCHMARK("Hausdorff Vec") {
        return utils::symmetricHausdorffDistance(distanceMatrix);
    };

    BENCHMARK("Hausdorff Sparse") {
        return symmetricHausdorffDistanceSparse(sparseDistanceMatrix);
    };

    fmt::println("");   // empty line for nicer formatting

}

void bm_shortest_path()
{
    // Define test data characteristics
    const int64_t numPoints     = 50000;
    const int64_t numDimensions = 3;
    const int64_t numNeighbors  = 120;
    const float noise           = 0.4f;

    const uint64_t seed         = 54321;

    std::vector<float> pointData, colors;
    utils::createSwissRole(pointData, colors, numPoints, noise, seed);
    const auto data = utils::Data(std::move(pointData), numDimensions);

    fmt::println("First three data points:");
    for(uint64_t i : std::views::iota(0ll, 3ll))
        utils::print(data.getValuesAt(i));

    // Helper to create random nodes
    std::random_device rd;
    std::mt19937_64 gen = std::mt19937_64(rd());
    gen.seed(seed);
    std::uniform_int_distribution<unsigned int> uniform_distribution = std::uniform_int_distribution<unsigned int>(0, static_cast<unsigned int>(data.getNumPoints() - 1));

    // Setup Knn
    auto knn = NearestNeighbors(data);
    knn.setCachingActive(false);

    NearestNeighborsSettings nns;
    nns.numNearestNeighbors = numNeighbors;
    nns.knnMetric = utils::KnnMetric::L2;
    nns.knnIndex = utils::KnnIndex::BruteForce;

    knn.compute(nns);
        
    const auto& knnGraph = knn.getKnnGraph();
    utils::BoostGraph bgraph = utils::createBoostGraph(knnGraph);
    const auto knnGraphView = knnGraph.getGraphView();

    int64_t startID = uniform_distribution(gen);
    int64_t endID = uniform_distribution(gen);

    fmt::println("Shortest path between {0} and {1}", startID, endID);

    {
        float g_dist(-1.f), BOOST_g_dist(-1.f), BOOST_SPH_dist(-1.f);
        std::vector<int64_t> geodesic;
        std::vector<float> distance_map;
        std::vector<uint64_t> predecessor_map;

        utils::astar(knnGraph, data, startID, endID, geodesic, g_dist);
        CHECK(g_dist >= 0);

        utils::astarBoost(bgraph, data, startID, endID, distance_map, predecessor_map, geodesic, BOOST_g_dist);
        CHECK(BOOST_g_dist >= 0);

        utils::astarBoost(knnGraphView, data, startID, endID, distance_map, predecessor_map, geodesic, BOOST_SPH_dist);
        CHECK(BOOST_g_dist >= 0);

        CHECK(g_dist - distanceL2(data.getData(), data.getNumDimensions(), startID, endID) >= -eps);
        CHECK_THAT(g_dist, Catch::Matchers::WithinRel(BOOST_g_dist, eps));
        CHECK_THAT(BOOST_SPH_dist, Catch::Matchers::WithinRel(BOOST_g_dist, eps));
    }

    // This library
    BENCHMARK("utils::astar") {
        float g_dist(-1.f);
        std::vector<int64_t> geodesic;
        utils::astar(knnGraph, data, startID, endID, geodesic, g_dist);
        return g_dist;
    };

    // Reference implementation
    BENCHMARK("utils::astarBoost") {
        float g_dist(-1.f);
        std::vector<int64_t> geodesic;
        std::vector<float> distance_map;
        std::vector<uint64_t> predecessor_map;
        utils::astarBoost(bgraph, data, startID, endID, distance_map, predecessor_map, geodesic, g_dist);
        return g_dist;
    };

    // Boost with own graph
    BENCHMARK("utils::astarBoost with SPH graph") {
        float g_dist(-1.f);
        std::vector<int64_t> geodesic;
        std::vector<float> distance_map;
        std::vector<uint64_t> predecessor_map;
        utils::astarBoost(knnGraphView, data, startID, endID, distance_map, predecessor_map, geodesic, g_dist);
        return g_dist;
    };

    fmt::println("");   // empty line for nicer formatting
}

void bm_symmetricGraph() {
    // define data and knn
    int64_t k(3), numPoints(24);

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;

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

    assert(knnGraph.isValid());

    auto knnGraphView = knnGraph.getGraphView();

    bool verbose = false;

    BENCHMARK("utils::symmetrizeGraph") {
        auto symGraph = utils::symmetrizeGraph(knnGraphView, verbose);
        return symGraph;
    };

    BENCHMARK("utils::symmetrizeGraphOld") {
        auto symGraph2 = utils::symmetrizeGraphOld(knnGraphView, verbose);
        return symGraph2;
    };

    BENCHMARK("utils::symmetrizeGraphEigen") {
        auto symGraph3 = utils::symmetrizeGraphEigen(knnGraphView, verbose);
        return symGraph3;
    };

    fmt::println("");   // empty line for nicer formatting
}

void bm_randomWalkSimilarities1() {

    const int verbosityLevel = 0;

    // Make sure the results are the same
    const size_t numRowsTest = 2000;
    const size_t numValuesInRowTest = 500;

    sph::SparseMatSPH test = createRandomSparseMatrix(numRowsTest, numValuesInRowTest);

    auto t0 = sph::utils::now();
    auto sims0 = utils::createSimilaritiesEigen(test, 0.f, verbosityLevel, 0);
    fmt::println("{} createSimilarities: {} ms (sequential)", numRowsTest, sph::utils::timeSince<std::chrono::milliseconds>(t0));

    auto t1 = sph::utils::now();
    auto sims1 = utils::createSimilaritiesEigen(test, 0.f, verbosityLevel, 1000);
    fmt::println("{} createSimilarities: {} ms (parallel)", numRowsTest, sph::utils::timeSince<std::chrono::milliseconds>(t1));

    auto t2 = sph::utils::now();
    auto sims2 = utils::createSimilaritiesSPH(test, 0.f, verbosityLevel);
    fmt::println("{} createSimilaritiesSPH: {} ms", numRowsTest, sph::utils::timeSince<std::chrono::milliseconds>(t2));

    REQUIRE(sims0.size() == sims1.size());
    REQUIRE(sims0.size() == sims2.size());

    for (size_t i = 0; i < sims0.size(); i++)
        REQUIRE(sims0[i].isApprox(sims1[i], 0.0001f));
    for (size_t i = 0; i < sims0.size(); i++)
        REQUIRE(sims0[i].isApprox(sims2[i], 0.0001f));
}

void bm_randomWalkSimilarities2() {

    const int verbosityLevel = 0;
    size_t numRows = 500;
    size_t numValuesInRow = 50;

    sph::SparseMatSPH input = createRandomSparseMatrix(numRows, numValuesInRow);

    BENCHMARK(fmt::format("utils::createSimilarities ({}, sequential)", numRows)) {
        auto sims = utils::createSimilaritiesEigen(input, 0.f, verbosityLevel, 0);
        return sims;
    };

    BENCHMARK(fmt::format("utils::createSimilarities ({}, parallel)", numRows)) {
        auto sims = utils::createSimilaritiesEigen(input, 0.f, verbosityLevel, 100);
        return sims;
    };

    BENCHMARK(fmt::format("utils::createSimilaritiesSPH ({})", numRows)) {
        auto sims = utils::createSimilaritiesSPH(input, 0.f, verbosityLevel);
        return sims;
    };

    fmt::println("");   // empty line for nicer formatting
}

void bm_randomWalkSimilarities3() {

    const int verbosityLevel = 0;

    fmt::println("createSimilarities seq vs par");
    for (const auto& multiplier : std::vector<size_t>{ 4, 6, 8, 10 })
    {
        const size_t numRows = 1000 * multiplier;
        const size_t numValuesInRow = 500;

        fmt::println("createSimilarities seq vs par: {}", numRows);

        sph::SparseMatSPH input = createRandomSparseMatrix(numRows, numValuesInRow);

        auto t0 = sph::utils::now();
        auto sims0 = utils::createSimilaritiesEigen(input, 0.f, verbosityLevel, 0);
        fmt::println("{} createSimilarities: {} ms (sequential)", numRows, sph::utils::timeSince<std::chrono::milliseconds>(t0));

        auto t1 = sph::utils::now();
        auto sims1 = utils::createSimilaritiesEigen(input, 0.f, verbosityLevel, 1'000);
        fmt::println("{} createSimilarities: {} ms (parallel)", numRows, sph::utils::timeSince<std::chrono::milliseconds>(t1));

        REQUIRE(sims0.size() == sims1.size());

        for (size_t i = 0; i < sims0.size(); i++)
            REQUIRE(sims0[i].isApprox(sims1[i], 0.0001f));
    }

    // Note: way too memory hungry
    //fmt::println("createSimilarities par");
    //{
    //    const size_t numRows = 50'000;
    //    const size_t numValuesInRow = 500;
    //    sph::SparseMatSPH input = createRandomSparseMatrix(numRows, numValuesInRow);

    //    for (const auto& multiplier : std::vector<size_t>{ 4, 6, 8, 10 })
    //    {
    //        const Eigen::Index blockSize = 1'000 * multiplier;

    //        fmt::println("createSimilarities par: {}, {} (size, blockSize)", numRows, blockSize);

    //        auto t0 = sph::utils::now();
    //        auto sims0 = utils::createSimilarities(input, 0.f, false, false, blockSize);
    //        fmt::println("{} createSimilarities: {} ms (parallel)", numRows, sph::utils::timeSince<std::chrono::milliseconds>(t0));
    //    }

    //}
}

void bm_randomWalkSimilarities4() {

    const int verbosityLevel = 0;

    fmt::println("bm_randomWalkSimilarities4 (+ conversion)");

    const size_t numRowsTest = 10000;
    const size_t numValuesInRowTest = 500;
    const size_t k = 30;
    const Eigen::Index blockSize = 1000;
    const float pruneVal = 0.f;

    fmt::println("createRandomSparseMatrix with {} numRowsTest...", numRowsTest);
    const sph::SparseMatSPH test = createRandomSparseMatrix(numRowsTest, numValuesInRowTest);

    fmt::println("createSimilaritiesEigen...");
    const auto t_Eigen = sph::utils::now();
    SparseMatHDI simsEigen(numRowsTest);
    const auto simsEigen_Full = utils::createSimilaritiesEigen(test, pruneVal, verbosityLevel, blockSize);
    SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(numRowsTest); ++i)
            utils::convertEigenSparseVecToHDILibSparseVec(simsEigen_Full[i], k, simsEigen[i], /*top=*/ false);
    fmt::println("createSimilaritiesEigen: {} ms (+ conversion)", sph::utils::timeSince<std::chrono::milliseconds>(t_Eigen));

    fmt::println("createSimilaritiesHDI...");
    const auto t_HDI = sph::utils::now();
    const auto simsHDI = utils::createSimilaritiesHDI(test, k, pruneVal, verbosityLevel, blockSize);
    fmt::println("createSimilaritiesHDI: {} ms", sph::utils::timeSince<std::chrono::milliseconds>(t_HDI));

    checkSameTwoMatricesHDI(simsEigen, simsHDI);

    // The following are very slow

    fmt::println("createSimilaritiesSPH...");
    const auto t_SPH = sph::utils::now();
    SparseMatHDI simsSPH(numRowsTest);
    const auto simsSPH_Full = utils::createSimilaritiesSPH(test, pruneVal, verbosityLevel);
    SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(numRowsTest); ++i)
            utils::convertEigenSparseVecToHDILibSparseVec(simsSPH_Full[i], k, simsSPH[i], /*top=*/ false);
    fmt::println("createSimilaritiesSPH: {} ms (+ conversion)", sph::utils::timeSince<std::chrono::milliseconds>(t_SPH));

    checkSameTwoMatricesHDI(simsEigen, simsSPH);
    checkSameTwoMatricesSPH(simsEigen_Full, simsSPH_Full);

    fmt::println("createSimilaritiesEigen0...");
    const auto t_Eigen0 = sph::utils::now();
    SparseMatHDI simsEigen0(numRowsTest);
    const auto simsEigen0_Full = utils::createSimilaritiesEigen(test, pruneVal, verbosityLevel, 0);
    SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(numRowsTest); ++i)
            utils::convertEigenSparseVecToHDILibSparseVec(simsEigen0_Full[i], k, simsEigen0[i], /*top=*/ false);
    fmt::println("createSimilaritiesEigen0: {} ms (+ conversion)", sph::utils::timeSince<std::chrono::milliseconds>(t_Eigen0));

    checkSameTwoMatricesHDI(simsEigen, simsEigen0);
    checkSameTwoMatricesSPH(simsEigen_Full, simsEigen0_Full);
}
