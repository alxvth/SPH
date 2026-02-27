/// Tests
/// Several test scenarios for SPH algorithms

// for info on testing see https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md#test-cases-and-sections
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>	

#include <cstdlib>
#include <vector>

#include <sph/utils/Data.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/TestData.hpp>

#include "AStarTest.hpp"
#include "HierarchyTest.hpp"
#include "UtilsBenchmark.hpp"
#include "UtilsTests.hpp"

constexpr int verbosity = 0;
bool expensive_tests = false;

using namespace sph;

/// A* shortest path
TEST_CASE("A* shortest path", "[ASTAR]") {
    Log::info("#### Test: A* shortest path ####");

    // Define test data characteristics
    const int64_t numPoints = 500000;
    const int64_t numDimensions = 3;
//    const std::vector<int64_t> numNeighbors = { 10, 15, 30, 60, 120 };
    const std::vector<int64_t> numNeighbors = { 120 };
    const int64_t numRuns = 100;

    // Noise might be different for each data set
    float noise = 0.4f;

    // Data
    std::vector<float> pointData;
    std::vector<float> colors;

    SECTION("Swiss Role") {
        Log::info("## Test: Swiss Role ##");
        Log::info("Data set with " + std::to_string(numPoints) + " data points");

        utils::createSwissRole(pointData, colors, numPoints, noise);

        const auto data = utils::Data(std::move(pointData), numDimensions);

        testAStar(numNeighbors, numRuns, data, verbosity);
    }

    SECTION("Swiss Role (Caching)") {
        Log::info("## Test: Swiss Role (Caching) ##");
        Log::info("Data set with " + std::to_string(numPoints) + " data points");

        utils::createSwissRole(pointData, colors, numPoints, noise);

        const auto data = utils::Data(std::move(pointData), numDimensions);

        testShortestPathCaching(numNeighbors, numRuns, data, verbosity);
    }

    SECTION("S Curve") {
        Log::info("## Test: S Curve ##");
        Log::info("Data set with " + std::to_string(numPoints) + " data points");

        noise = 0.1f; 
        utils::createSCurve(pointData, colors, numPoints, noise);

        const auto data = utils::Data(std::move(pointData), numDimensions);

        testAStar(numNeighbors, numRuns, data, verbosity);
    }

    SECTION("S Curve (Caching)") {
        Log::info("## Test: S Curve (Caching) ##");
        Log::info("Data set with " + std::to_string(numPoints) + " data points");

        noise = 0.1f; 
        utils::createSCurve(pointData, colors, numPoints, noise);

        const auto data = utils::Data(std::move(pointData), numDimensions);

        testShortestPathCaching(numNeighbors, numRuns, data, verbosity);
    }

}

/// Image Hierarchy
TEST_CASE("Image Hierarchy", "[IMAGE][HIERARCHY]") {
    Log::info("#### Test: Image Hierarchy ####");

    Log::info("## Test: Hierarchy Traversal ##");
    testHierarchyTraversal();

    Log::info("## Test: Pixel neighbors ##");
    testPixelNeighbors();

    Log::info("## Test: Knn Overlap ##");
    testKnnOverlap();

    Log::info("## Test: Image Hierarchy Overlap ##");
    testImageHierarchyOverlap();

    Log::info("## Test: Image Hierarchy Walks ##");
    testImageHierarchyWalks();

    Log::info("## Test: Image Hierarchy with non-rectangular image input (Overlap) ##");
    testNonRectImageOverlap();

    Log::info("## Test: Image Hierarchy with non-rectangular image input (Walks) ##");
    testNonRectImageWalk();

    Log::info("## Test: Node merging (synth) ##");
    testMergeNodesSynth();

    Log::info("## Test: Node merging (data graph) ##");
    testMergeNodesGraph();
}

/// Utils
TEST_CASE("Utility functions", "[UTILS]") {
    Log::info("#### Test: Utility functions ####");

    Log::info("## Test: testComputeQuantile ##");
    testComputeQuantile();

    Log::info("## Test: testClampValues ##");
    testClampValues();

    Log::info("## Test: testLabelGraphComponents ##");
    testLabelGraphComponents();

    Log::info("## Test: testL2 ##");
    testL2();

    Log::info("## Test: testSymmetrizeKnnGraph ##");
    testSymmetrizeKnnGraph();

    Log::info("## Test: testConnectingComponentsKnnGraph ##");
    testConnectingComponentsKnnGraph();

    Log::info("## Test: testGraphAccess() ##");
    testGraphAccess();

    Log::info("## Test: testMath() ##");
    testMath();

    Log::info("## Test: testMaxSizeDeque() ##");
    testMaxSizeDeque();

    Log::info("## Test: testEigenSparseMatrix() ##");
    testEigenSparseMatrix();

    Log::info("## Test: testSymmetrizeDistUMAP() ##");
    testSymmetrizeDistUMAP();

    if (expensive_tests)
    {
        Log::info("## Test: testNodeMerging() ##");
        testNodeMerging();
    }
}

TEST_CASE("File IO functions", "[FILEIO]") {
    Log::info("#### Test: File IO functions ####");

    Log::info("## Test: testIOCompressedSparseMatHDIBinary() ##");
    testIOCompressedSparseMatHDIBinary();

    Log::info("## Test: testIOCompressedSparseMatSPHBinary() ##");
    testIOCompressedSparseMatSPHBinary();

    Log::info("## Test: testIOCompressedVecBinary() ##");
    testIOCompressedVecBinary();
}

TEST_CASE("GPU functions", "[GPU]") {
    Log::info("#### Test: GPU functions ####");

}

/// Benchmarking
TEST_CASE("Utility functions benchmarking", "[BENCHMARK]") {
    Log::info("## Benchmark: bm_hausdorff ##");
    bm_hausdorff();

    //Log::info("## Benchmark: bm_shortest_path ##");
    //bm_shortest_path();

    //Log::info("## Benchmark: bm_symmetricGraph ##");
    //bm_symmetricGraph();

    //Log::info("## Benchmark: bm_randomWalkSimilarities1 ##");
    //bm_randomWalkSimilarities1();

    //Log::info("## Benchmark: bm_randomWalkSimilarities2 ##");
    //bm_randomWalkSimilarities2();

    //Log::info("## Benchmark: bm_randomWalkSimilarities3 ##");
    //bm_randomWalkSimilarities3();

    Log::info("## Benchmark: bm_randomWalkSimilarities4 ##");
    bm_randomWalkSimilarities4();

}

int main(int argc, char* argv[]) {
    Catch::Session session;

    auto cli = session.cli()
        | Catch::Clara::Opt(expensive_tests)
        ["--run_expensive_tests"]
        ("Runs expensive tests.");

    session.cli(cli);

    if (expensive_tests) {
        Log::info("Test option: --run_expensive_tests");
    }

    if (argc < 2)
    {
        session.configData().testsOrTags.push_back("[IMAGE],");
        session.configData().testsOrTags.push_back("[HIERARCHY],");
        session.configData().testsOrTags.push_back("[UTILS],");
        //session.configData().testsOrTags.push_back("[FILEIO],");
        //session.configData().testsOrTags.push_back("[ASTAR],");
        //session.configData().testsOrTags.push_back("[BENCHMARK],");
    }

//#ifndef NDEBUG
//    Log::set_level(spdlog::level::trace);
//#endif // !NDEBUG

    int numFailed = session.run(argc, argv);

    if (numFailed > 0)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
