#include "HierarchyTest.hpp"

#include <sph/ImageHierarchy.hpp>
#include <sph/NearestNeighbors.hpp>
#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/EvalIO.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/Hierarchy.hpp>
#include <sph/utils/ImageHelper.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/Settings.hpp>
#include <sph/utils/Similarities.hpp>
#include <sph/utils/SparseMatrixAlgorithms.hpp>

#include <catch2/catch_test_macros.hpp>	
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>	
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/src/SparseCore/SparseVector.h>

constexpr float eps = 0.0001f;

using namespace sph;
namespace fs = std::filesystem;

void testHierarchyTraversal()
{
    utils::Hierarchy h;
    h.children.resize(2);                                        // 3 level - the base level does not have children
    h.children[1] = {     {0,1,2},           {3,4} };            // On the top level (2) nodes there are 2
    h.children[0] = { {0,1}, {2,3,4}, {5,6,7}, {8,9}, {10,11} }; // 5 mid level (1) and 12 low level (0) nodes

    std::vector<uint64_t> redIDs;

    // level 1, component 0
    utils::ComponentID id{ 1,0 };
    h.getRepresentedDataPoints(id, redIDs);
    REQUIRE(redIDs == vui64{0, 1});
    
    // level 2, component 1
    id = { 2,1 };
    h.getRepresentedDataPoints(id, redIDs);
    REQUIRE(redIDs == vui64{8, 9, 10 ,11 });
    
    // level 2, component 0
    id = { 2,0 };
    h.getRepresentedDataPoints(id, redIDs);
    REQUIRE(redIDs == vui64{0, 1, 2, 3, 4, 5, 6, 7});
    
    // level 0, component 5
    id = { 0,5 };
    h.getRepresentedDataPoints(id, redIDs);
    REQUIRE(redIDs == vui64{5});
    
    // level 0, component 0
    id = { 0,0 };
    h.getRepresentedDataPoints(id, redIDs);
    REQUIRE(redIDs == vui64{0});

    // level 0, component 11
    id = { 0,11 };
    h.getRepresentedDataPoints(id, redIDs);
    REQUIRE(redIDs == vui64{11});
}

void testPixelNeighbors()
{
    // image data
    uint64_t rows(4), cols(6);

    // unused variables (in this test)
    uint64_t numDims(1);

    utils::Data data;
    data.numPoints = rows * cols;
    data.numDimensions = numDims;
    data.dataVec.resize(rows * cols * numDims);

    utils::Graph knnGraph;    // not used in this test

    ImageHierarchy ih(knnGraph, data, rows, cols);

    // 4 neighbors
    ih.setNeighConnection(utils::NeighConnection::FOUR);

    std::vector<uint64_t> neighIDs;

    auto getPixelNeighborIDs = [&ih, &neighIDs](uint64_t id) -> void {
        utils::pixelNeighborIDs(ih.getNumCols(), ih.getNumRows(), ih.getConnectedNeighbors(), id, neighIDs);
    };
    
    getPixelNeighborIDs(0);
    REQUIRE(neighIDs == vui64{1, 6});

    getPixelNeighborIDs(5);
    REQUIRE(neighIDs == vui64{11, 4});

    getPixelNeighborIDs(18);
    REQUIRE(neighIDs == vui64{12, 19});

    getPixelNeighborIDs(23);
    REQUIRE(neighIDs == vui64{17, 22});

    getPixelNeighborIDs(6);
    REQUIRE(neighIDs == vui64{0, 7, 12});

    getPixelNeighborIDs(17);
    REQUIRE(neighIDs == vui64{11, 23, 16});

    getPixelNeighborIDs(7);
    REQUIRE(neighIDs == vui64{1, 8, 13, 6});

    getPixelNeighborIDs(15);
    REQUIRE(neighIDs == vui64{9, 16, 21, 14});

    // 8 neighbors
    ih.setNeighConnection(utils::NeighConnection::EIGHT);

    getPixelNeighborIDs(0);
    REQUIRE(neighIDs == vui64{1, 6, 7});

    getPixelNeighborIDs(5);
    REQUIRE(neighIDs == vui64{4, 10, 11});

    getPixelNeighborIDs(18);
    REQUIRE(neighIDs == vui64{12, 13, 19});

    getPixelNeighborIDs(23);
    REQUIRE(neighIDs == vui64{16, 17, 22});

    getPixelNeighborIDs(6);
    REQUIRE(neighIDs == vui64{0, 1, 7, 12, 13});

    getPixelNeighborIDs(17);
    REQUIRE(neighIDs == vui64{10, 11, 16, 22, 23});

    getPixelNeighborIDs(7);
    REQUIRE(neighIDs == vui64{0, 1, 2, 6, 8, 12, 13, 14});

    getPixelNeighborIDs(15);
    REQUIRE(neighIDs == vui64{8, 9, 10, 14, 16, 20, 21, 22});

}

void testKnnOverlap() 
{
    // define data and knn
    int64_t k(3), numPoints(24); 
    
    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;
    knnGraph.knnDistances.resize(numPoints * k, 0);    // not used in this test

    knnGraph.knnIndices= {
        0,   1,  2,      1, 8,  9,      2,  4, 20,      3,  7, 15,      4, 16, 18,      5, 16,  0,
        6,   3,  8,      7, 8, 10,      8,  4, 21,      9, 21,  0,     10,  9,  3,     11, 10, 12,
        12, 11, 22,     13, 1,  3,     14,  9, 18,     15,  2, 11,     16,  6, 15,     17, 16, 18,
        18,  1, 19,     19, 2, 23,     20, 23, 21,     21, 23, 15,     22, 17, 14,     23, 13, 12,
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    auto knnGraphView = knnGraph.getGraphView();

    // image data
    //int64_t rows(4), cols(6);

    utils::Hierarchy h;
    // 3 level - the base level does not have children
    h.children.resize(2);                                        
    // On the top level (2) nodes there are 2
    h.children[1] = { {0,1,2}, {3,4}, {5, 8}, {6, 7} };
    // 9 mid level (1) and 24 low level (0) nodes - the image size
    h.children[0] = { {0,1}, {2,3,4}, {5,6,7}, {8,9}, {10,11}, {12, 13, 14, 15, 16}, {17, 20, 22}, {18, 21}, {19, 23} };

    std::vector<uint64_t> unionNeighbors1, unionNeighbors2;
    uint64_t overlapSize = 0;

    utils::ComponentID id1{ 1,0 };
    utils::ComponentID id2{ 2,0 };
    overlapSize = utils::representedOverlap(h, &knnGraphView, id1, id2, unionNeighbors1, unionNeighbors2);
    REQUIRE(unionNeighbors1 == vui64{0, 1, 2,                8, 9});
    REQUIRE(unionNeighbors2 == vui64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 18, 20});
    REQUIRE(overlapSize == 5);

    id1 = { 1,8 };
    id2 = { 1,0 };
    overlapSize = utils::representedOverlap(h, &knnGraphView, id1, id2, unionNeighbors1, unionNeighbors2);
    REQUIRE(unionNeighbors1 == vui64{2, 12, 13, 19, 23});
    REQUIRE(unionNeighbors2 == vui64{0, 1, 2, 8, 9});
    REQUIRE(overlapSize == 1);

    id1 = { 2,1 };
    id2 = { 2,2 };
    overlapSize = utils::representedOverlap(h, &knnGraphView, id1, id2, unionNeighbors1, unionNeighbors2);
    REQUIRE(unionNeighbors1 == vui64{0, 3, 4, 8, 9, 10, 11, 12, 21});
    REQUIRE(unionNeighbors2 == vui64{1, 2, 3, 6, 9, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23});
    REQUIRE(overlapSize == 4);

    id1 = { 0,0 };
    id2 = { 0,1 };
    overlapSize = utils::representedOverlap(h, &knnGraphView, id1, id2, unionNeighbors1, unionNeighbors2);
    REQUIRE(unionNeighbors1 == vui64{0, 1, 2});
    REQUIRE(unionNeighbors2 == vui64{1, 8, 9});
    REQUIRE(overlapSize == 1);

    id1 = { 0,23 };
    id2 = { 1,1 };
    overlapSize = utils::representedOverlap(h, &knnGraphView, id1, id2, unionNeighbors1, unionNeighbors2);
    REQUIRE(unionNeighbors1 == vui64{12, 13, 23});
    REQUIRE(unionNeighbors2 == vui64{2, 3, 4, 7, 15, 16, 18, 20});
    REQUIRE(overlapSize == 0);

}

void testImageHierarchyOverlap()
{
    // define data and knn

    // image data
    int64_t rows(4), cols(4);
    auto numPoints = rows * cols;
    int64_t k(4), numDims(2);

    utils::Data data;
    data.numPoints = numPoints;
    data.numDimensions = numDims;
    data.dataVec.resize(numPoints * numDims, -1); // not used in this test

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;
    knnGraph.knnDistances.resize(numPoints * k, 0.f);    // not used in this test

    // pre-defined neighbors
    knnGraph.knnIndices = {
     0,  1,  2,  4,      1,  2,  3,  4,      2,  3,  4,  5,     3,  2,  5,  6,
     4,  8, 12,  3,      5,  6,  7, 15,      6, 10,  7,  1,     7, 11,  0, 15,
     8, 12,  2,  3,      9,  5,  0, 15,     10,  9,  0,  7,    11, 15, 12,  9,
    12,  8,  2,  3,     13, 14,  7,  2,     14, 13,  2, 11,    15, 11, 12,  5,
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    ImageHierarchy ih(knnGraph, data, rows, cols);
    ih.setCachingActive(false);
    ih.setComponentSim(utils::ComponentSim::NEIGH_OVERLAP);
    ih.setNeighConnection(utils::NeighConnection::FOUR);
    ih.setMergeMultipleComponents(false);
    ih.setUsePercentile(false);
    ih.compute();

    auto& h = ih.getHierarchy();
    const auto numLevels = h.getNumLevels();

    REQUIRE(numLevels - 1 == h.parents.size());
    REQUIRE(numLevels - 1 == h.children.size());
    REQUIRE(numLevels - 1 == h.spatialNeighbors.size());
    REQUIRE(numLevels == h.pixelComponents.size());

    std::cout << "\n";

    /*
    Level: 0
    0   1  2  3
    4   5  6  7
    8   9 10 11
    12 13 14 15
    */
    uint64_t l = 0;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    /*
    Level: 1
     0  0  0  0
     1  2  2  3
     1  2  2  3
     1  4  4  3
     */
    l = 1;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 1, 2, 2, 3, 1, 2,  2,  3,  1,  4,  4,  3});

    REQUIRE(ih.getSpatialNeighbors({ 1, 0 }) == vui64{ 1, 2, 3});
    REQUIRE(ih.getSpatialNeighbors({ 1, 1 }) == vui64{ 0, 2, 4});
    REQUIRE(ih.getSpatialNeighbors({ 1, 2 }) == vui64{ 0, 1, 3, 4});
    REQUIRE(ih.getSpatialNeighbors({ 1, 3 }) == vui64{ 0, 2, 4});
    REQUIRE(ih.getSpatialNeighbors({ 1, 4 }) == vui64{ 1, 2, 3});

    /*
    Level: 2
     0  0  0  0
     0  1  1  1
     0  1  1  1
     0  1  1  1
     */
    l = 2;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,  1,  1,  0,  1,  1,  1});

    REQUIRE(ih.getSpatialNeighbors({ 2, 0 }) == vui64{ 1 });
    REQUIRE(ih.getSpatialNeighbors({ 2, 1 }) == vui64{ 0 });

    /*
    Level: 3
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     */
    l = 3;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0});

    REQUIRE(h.parentsOn(0).size() == h.numComponentsOn(0));
    REQUIRE(h.parentsOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.parentsOn(2).size() == h.numComponentsOn(2));

    REQUIRE(h.parentsOn(0) == vui64{ 0, 0, 0, 0, 1, 2, 2, 3, 1, 2, 2, 3, 1, 4, 4, 3});
    REQUIRE(h.parentsOn(1) == vui64{ 0, 0, 1, 1, 1});
    REQUIRE(h.parentsOn(2) == vui64{ 0, 0});

    REQUIRE(h.childrenOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.childrenOn(2).size() == h.numComponentsOn(2));
    REQUIRE(h.childrenOn(3).size() == h.numComponentsOn(3));

    REQUIRE(h.childrenOn(1) == vvui64{ vui64{0, 1, 2, 3}, vui64{4, 8, 12}, vui64{5, 6, 9, 10}, vui64{7, 11, 15}, vui64{13, 14}});
    REQUIRE(h.childrenOn(2) == vvui64{ vui64{0, 1}, vui64{2, 3, 4} });
    REQUIRE(h.childrenOn(3) == vvui64{ vui64{0, 1} });

    REQUIRE(h.spatialNeighborsOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.spatialNeighborsOn(2).size() == h.numComponentsOn(2));
    REQUIRE(h.spatialNeighborsOn(3).size() == h.numComponentsOn(3));

    REQUIRE(h.spatialNeighborsOn(1) == vvui64{ vui64{1, 2, 3}, vui64{0, 2, 4}, vui64{0, 1, 3, 4}, vui64{0, 2, 4}, vui64{1, 2, 3}});
    REQUIRE(h.spatialNeighborsOn(2) == vvui64{ vui64{1}, vui64{0} });
    REQUIRE(h.spatialNeighborsOn(3) == vvui64{ {} });

}

void testImageHierarchyWalks()
{
    // define data and knn

    // image data
    int64_t rows(4), cols(4);
    auto numPoints = rows * cols;
    int64_t k(4), numDims(2);

    utils::Data data;
    data.numPoints = numPoints;
    data.numDimensions = numDims;
    data.dataVec.resize(numPoints * numDims, -1); // not used in this test

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;

    // pre-defined neighbors
    knnGraph.knnIndices = {
     0,  1,  2,  4,      1,  2,  3,  4,      2,  3,  4,  5,     3,  2,  5,  6,
     4,  8, 12,  3,      5,  6,  7, 15,      6, 10,  7,  1,     7, 11,  0, 15,
     8, 12,  2,  3,      9,  5,  0, 15,     10,  9,  0,  7,    11, 15, 12,  9,
    12,  8,  2,  3,     13, 14,  7,  2,     14, 13,  2, 11,    15, 11, 12,  5,
    };

    knnGraph.knnDistances = {
     0.f, .1f ,  .2f,  .4f,     0.f, .2f , .3f,  .4f,     0.f,  .3f,  .4f,  .5f,    0.f, .2f ,  .5f ,  .6f,
     0.f, .8f , 1.2f, 3.f ,     0.f, .6f , .7f, 1.5f,     0.f, .10f,  .7f, 1.f ,    0.f, .11f,  .4f , 1.5f,
     0.f, .12f,  .2f,  .3f,     0.f, .5f , .6f, 1.5f,     0.f, .9f , 1.2f, 7.f ,    0.f, .15f, 1.2f , 9.f ,
     0.f, .8f , 2.f , 3.f ,     0.f, .14f, .7f, 2.f ,     0.f, .13f,  .2f, 1.1f,    0.f, .11f,  .12f,  .5f,
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    // ensure the distances make sense, i.e. they are in increasing order
    for (int64_t i = 0; i < knnGraph.getNumPoints(); i++)
        for (int64_t j = 0; j < k - 1; j++)
            REQUIRE(knnGraph.getDistanceN(i, j) <= knnGraph.getDistanceN(i, j + 1));

    ImageHierarchy ih(knnGraph, data, rows, cols);
    ih.setCachingActive(false);
    ih.setComponentSim(utils::ComponentSim::NEIGH_WALKS);
    ih.setNeighConnection(utils::NeighConnection::FOUR);
    ih.setRandomWalkHandling(utils::RandomWalkHandling::MERGE_RW_NEW_WALKS);
    ih.compute();

    auto& h = ih.getHierarchy();

    std::cout << "\n";
    std::cout << "Image components" << "\n";
    for (size_t i = 0; i < h.getNumLevels(); i++)
    {
        std::cout << "Level: " << i << "\n";
        utils::printImageComponents(ih, i);
    }

    std::cout << "Random Walk Similarities" << "\n";
    for (size_t i = 0; i < h.getNumLevels(); i++)
    {
        std::cout << "Level: " << i << "\n";

        if(i > 0)
            for (uint64_t j = 0; j < h.numComponents[i]; j++)
            {
                std::cout << j << " merged from ";
                utils::print(h.childrenOn(i)[j]);
            }

        const auto& similarities = h.randomWalks[i];

        std::cout << "\n";
        utils::printSparseMatrixAsDense(similarities, true);

        for (size_t j = 0; j < similarities.size(); j++)
        {
            float simSum = similarities[j].sum();
            REQUIRE_THAT(simSum, Catch::Matchers::WithinAbs(1, 0.001));
        }
           

        if (i < h.getNumLevels() - 1)
        {
            for (uint64_t j = 0; j < h.numComponents[i]; j++)
                std::cout << j << " merges into " << h.parentsOn(i)[j] << "\n";
            std::cout << "\n";
        }
    }

    std::cout << "\n";
}

void testNonRectImageWalk()
{
    // define data and knn

    // image data
    int64_t rows(6), cols(4);
    auto numPoints = rows * cols;
    int64_t k(4), numDims(2);

    utils::Data data;
    data.numPoints = numPoints;
    data.numDimensions = numDims;
    data.dataVec.resize(numPoints * numDims, -1); // not used in this test

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;

    // pre-defined neighbors
    knnGraph.knnIndices = {
     0,  1,  2,  4,      1,  2,  3,  4,      2,  3,  4,  5,     3,  2,  5,  6,
     4,  8, 12,  3,      5,  6,  7, 15,      6, 10,  7,  1,     7, 11,  0, 15,
     8, 12,  2,  3,      9,  5,  0, 15,     10,  9,  0,  7,    11, 15, 12,  9,
    12,  8,  2,  3,     13, 14,  7,  2,     14, 13,  2, 11,    15, 11, 12,  5,
    16, 17, 18, 20,     17, 16, 19, 18,     18, 19, 16, 17,    19, 18, 17, 16,
    20, 21, 22, 16,     21, 20, 22, 23,     22, 23, 21, 20,    23, 22, 20,  0,
    };

    knnGraph.knnDistances = {
     0.f, 1.0f,  2.0f,   4.0f,     0.f, 2.0f, 3.0f,  4.0f,     0.f, 3.0f,  4.0f,  5.0f,    0.f, 2.0f,  5.0f,  6.0f,
     0.f, 8.0f, 12.0f,  30.0f,     0.f, 0.1f, 7.0f, 15.0f,     0.f, 1.0f,  7.0f, 10.0f,    0.f, 1.1f,  4.0f, 15.0f,
     0.f, 1.1f,  2.0f,   3.0f,     0.f, 6.0f, 7.0f, 15.0f,     0.f, 9.0f, 10.2f, 70.0f,    0.f, 1.5f, 10.2f, 90.0f,
     0.f, 8.0f, 20.0f , 30.0f,     0.f, 1.4f, 7.0f, 20.0f,     0.f, 1.3f,  2.0f, 10.1f,    0.f, 1.1f,  1.2f,  5.0f,
     0.f, 9.0f, 60.0f , 70.0f,     0.f, 6.4f, 10.0f, 12.0f,     0.f, 3.0f,  8.0f, 25.0f,    0.f, 6.7f,  9.4f, 10.9f,
     0.f, 2.0f,  4.0f , 60.0f,     0.f, 1.5f, 8.0f, 50.0f,     0.f, 2.4f, 21.0f, 31.0f,    0.f, 1.8f,  4.2f,  6.4f,
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    // ensure the distances make sense, i.e. they are in increasing order
    for (int64_t i = 0; i < knnGraph.getNumPoints(); i++)
        for (int64_t j = 0; j < k - 1; j++)
            REQUIRE(knnGraph.getDistanceN(i, j) <= knnGraph.getDistanceN(i, j + 1));

    ImageHierarchy ih(knnGraph, data, rows, cols);
    ih.setCachingActive(false);
    ih.setComponentSim(utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP);
    ih.setNeighConnection(utils::NeighConnection::FOUR);
    ih.setMergeMultipleComponents(false);
    ih.setUsePercentile(false);
    ih.setRandomWalkHandling(utils::RandomWalkHandling::MERGE_RW_NEW_WALKS);
    ih.setNormInputDistances(utils::NormalizationScheme::LINEAR);   // NORMAL weighting would basically set large distances to 0 similarity here due to small perplexity
    utils::RandomWalkSettings rws;
    rws.singleWalkLength = 3;
    rws.numRandomWalks = 10;
    rws.importanceWeighting = utils::ImportanceWeighting::CONSTANT;
    rws.parallel = false;   // Otherwise the handpicked knnDistances won't result in the expected merging since each thread uses it's own uniform_real_distribution
    ih.setRandomWalkSettings(rws);
    ih.setVerbose(true);
    ih.compute();

    auto& h = ih.getHierarchy();

    const auto numLevels = h.getNumLevels();

    REQUIRE(numLevels - 1 == h.parents.size());
    REQUIRE(numLevels - 1 == h.children.size());
    REQUIRE(numLevels - 1 == h.spatialNeighbors.size());
    REQUIRE(numLevels == h.pixelComponents.size());

    std::cout << "\n";

    REQUIRE(numLevels == 4); // three abstractions and one data

    /*
    Level: 0
    0  1  2  3
    4  5  6  7
    8  9 10 11
    12 13 14 15
    16 17 18 19
    20 21 22 23
    */
    uint64_t l = 0;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });

    /*
    Level: 1
     0  0  0  0
     1  2  2  3
     1  2  2  3
     1  4  4  3
     5  5  6  6
     7  7  8  8
     */
    l = 1;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 1, 2, 2, 3, 1, 2, 2, 3, 1, 4, 4, 3, 5, 5, 6, 6, 7, 7, 8, 8 });

    REQUIRE(ih.getSpatialNeighbors({ 1, 0 }) == vui64{ 1, 2, 3 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 1 }) == vui64{ 0, 2, 4, 5 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 2 }) == vui64{ 0, 1, 3, 4 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 3 }) == vui64{ 0, 2, 4, 6 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 4 }) == vui64{ 1, 2, 3, 5, 6 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 5 }) == vui64{ 1, 4, 6, 7 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 6 }) == vui64{ 3, 4, 5, 8 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 7 }) == vui64{ 5, 8 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 8 }) == vui64{ 6, 7 });

    /*
    Level: 2
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     1  1  1  1
     2  2  2  2
     */
    l = 2;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2, });

    REQUIRE(ih.getSpatialNeighbors({ 2, 0 }) == vui64{ 1 });
    REQUIRE(ih.getSpatialNeighbors({ 2, 1 }) == vui64{ 0, 2 });
    REQUIRE(ih.getSpatialNeighbors({ 2, 2 }) == vui64{ 1 });

    /*
    Level: 3
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     */
    l = 3;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

    REQUIRE(ih.getSpatialNeighbors({ 3, 0 }) == vui64{ });

    // All levels
    REQUIRE(h.parentsOn(0).size() == h.numComponentsOn(0));
    REQUIRE(h.parentsOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.parentsOn(2).size() == h.numComponentsOn(2));

    REQUIRE(h.parentsOn(0) == vui64{ 0, 0, 0, 0, 1, 2, 2, 3, 1, 2, 2, 3, 1, 4, 4, 3, 5, 5, 6, 6, 7, 7, 8, 8 });
    REQUIRE(h.parentsOn(1) == vui64{ 0, 0, 0, 0, 0, 1, 1, 2, 2 });
    REQUIRE(h.parentsOn(2) == vui64{ 0, 0, 0 });

    REQUIRE(h.childrenOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.childrenOn(2).size() == h.numComponentsOn(2));
    REQUIRE(h.childrenOn(3).size() == h.numComponentsOn(3));

    REQUIRE(h.childrenOn(1) == vvui64{ vui64{0, 1, 2, 3}, vui64{4, 8, 12}, vui64{5, 6, 9, 10}, vui64{7, 11, 15}, vui64{13, 14}, vui64{16, 17}, vui64{18, 19}, vui64{20, 21}, vui64{22, 23} });
    REQUIRE(h.childrenOn(2) == vvui64{ vui64{0, 1, 2, 3, 4}, vui64{5, 6}, vui64{7, 8} });
    REQUIRE(h.childrenOn(3) == vvui64{ vui64{0, 1, 2} });

    REQUIRE(h.spatialNeighborsOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.spatialNeighborsOn(2).size() == h.numComponentsOn(2));
    REQUIRE(h.spatialNeighborsOn(3).size() == h.numComponentsOn(3));

    REQUIRE(h.spatialNeighborsOn(1) == vvui64{ vui64{1, 2, 3}, vui64{0, 2, 4, 5}, vui64{0, 1, 3, 4}, vui64{0, 2, 4, 6}, vui64{1, 2, 3, 5, 6}, vui64{1, 4, 6, 7}, vui64{3, 4, 5, 8}, vui64{5, 8}, vui64{6, 7} });
    REQUIRE(h.spatialNeighborsOn(2) == vvui64{ vui64{1}, vui64{0, 2}, vui64{1} });
    REQUIRE(h.spatialNeighborsOn(3) == vvui64{ vui64{ } });

}


void testNonRectImageOverlap()
{
    // define data and knn

    // image data
    int64_t rows(6), cols(4);
    auto numPoints = rows * cols;
    int64_t k(4), numDims(2);

    utils::Data data;
    data.numPoints = numPoints;
    data.numDimensions = numDims;
    data.dataVec.resize(numPoints * numDims, -1); // not used in this test

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;
    knnGraph.knnDistances.resize(numPoints * k, 0.f);    // not used in this test

    // pre-defined neighbors
    knnGraph.knnIndices = {
     0,  1,  2,  4,      1,  2,  3,  4,      2,  3,  4,  5,     3,  2,  5,  6,
     4,  8, 12,  3,      5,  6,  7, 15,      6, 10,  7,  1,     7, 11,  0, 15,
     8, 12,  2,  3,      9,  5,  0, 15,     10,  9,  0,  7,    11, 15, 12,  9,
    12,  8,  2,  3,     13, 14,  7,  2,     14, 13,  2, 11,    15, 11, 12,  5,
    16, 17, 18, 20,     17, 16, 19, 18,     18, 19, 16, 17,    19, 18, 17, 16,
    20, 21, 22, 16,     21, 20, 22, 23,     22, 23, 21, 20,    23, 22, 20,  0,
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    ImageHierarchy ih(knnGraph, data, rows, cols);
    ih.setCachingActive(false);
    ih.setComponentSim(utils::ComponentSim::NEIGH_OVERLAP);
    ih.setNeighConnection(utils::NeighConnection::FOUR);
    ih.setRandomWalkHandling(utils::RandomWalkHandling::MERGE_RW_NEW_WALKS);
    ih.setMergeMultipleComponents(false);
    ih.setUsePercentile(false);
    ih.compute();

    auto& h = ih.getHierarchy();

    const auto numLevels = h.getNumLevels();

    REQUIRE(numLevels - 1 == h.parents.size());
    REQUIRE(numLevels - 1 == h.children.size());
    REQUIRE(numLevels - 1 == h.spatialNeighbors.size());
    REQUIRE(numLevels == h.pixelComponents.size());

    std::cout << "\n";

    /*
    Level: 0
    0  1  2  3
    4  5  6  7
    8  9 10 11
    12 13 14 15
    16 17 18 19
    20 21 22 23
    */
    uint64_t l = 0;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });

    /*
    Level: 1
     0  0  0  0
     1  2  2  3
     1  2  2  3
     1  4  4  3
     5  5  5  5
     6  6  6  6
     */
    l = 1;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 1, 2, 2, 3, 1, 2, 2, 3, 1, 4, 4, 3, 5, 5, 5, 5, 6, 6, 6, 6 });

    REQUIRE(ih.getSpatialNeighbors({ 1, 0 }) == vui64{ 1, 2, 3 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 1 }) == vui64{ 0, 2, 4, 5 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 2 }) == vui64{ 0, 1, 3, 4 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 3 }) == vui64{ 0, 2, 4, 5 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 4 }) == vui64{ 1, 2, 3, 5 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 5 }) == vui64{ 1, 3, 4, 6 });
    REQUIRE(ih.getSpatialNeighbors({ 1, 6 }) == vui64{ 5 });

    /*
    Level: 2
     0  0  0  0
     0  1  1  1
     0  1  1  1
     0  1  1  1
     2  2  2  2
     2  2  2  2
     */
    l = 2;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 });

    REQUIRE(ih.getSpatialNeighbors({ 2, 0 }) == vui64{ 1, 2 });
    REQUIRE(ih.getSpatialNeighbors({ 2, 1 }) == vui64{ 0, 2 });
    REQUIRE(ih.getSpatialNeighbors({ 2, 2 }) == vui64{ 0, 1 });

    /*
    Level: 3
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     0  0  0  0
     */
    l = 3;
    std::cout << std::format("Component layout on level {} is:\n", l);
    utils::printImageComponents(ih, l);

    REQUIRE(h.pixelComponentsOn(l) == vui64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

    REQUIRE(h.parentsOn(0).size() == h.numComponentsOn(0));
    REQUIRE(h.parentsOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.parentsOn(2).size() == h.numComponentsOn(2));

    REQUIRE(h.parentsOn(0) == vui64{ 0, 0, 0, 0, 1, 2, 2, 3, 1, 2, 2, 3, 1, 4, 4, 3, 5, 5, 5, 5, 6, 6, 6, 6 });
    REQUIRE(h.parentsOn(1) == vui64{ 0, 0, 1, 1, 1, 2, 2 });
    REQUIRE(h.parentsOn(2) == vui64{ 0, 0, 0 });

    REQUIRE(h.childrenOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.childrenOn(2).size() == h.numComponentsOn(2));
    REQUIRE(h.childrenOn(3).size() == h.numComponentsOn(3));

    REQUIRE(h.childrenOn(1) == vvui64{ vui64{0, 1, 2, 3}, vui64{4, 8, 12}, vui64{5, 6, 9, 10}, vui64{7, 11, 15}, vui64{13, 14}, vui64{16, 17, 18, 19}, vui64{20, 21, 22, 23} });
    REQUIRE(h.childrenOn(2) == vvui64{ vui64{0, 1}, vui64{2, 3, 4}, vui64{5, 6} });
    REQUIRE(h.childrenOn(3) == vvui64{ vui64{0, 1, 2} });

    REQUIRE(h.spatialNeighborsOn(1).size() == h.numComponentsOn(1));
    REQUIRE(h.spatialNeighborsOn(2).size() == h.numComponentsOn(2));
    REQUIRE(h.spatialNeighborsOn(3).size() == h.numComponentsOn(3));

    REQUIRE(h.spatialNeighborsOn(1) == vvui64{ vui64{1, 2, 3}, vui64{0, 2, 4, 5}, vui64{0, 1, 3, 4}, vui64{0, 2, 4, 5}, vui64{1, 2, 3, 5}, vui64{1, 3, 4, 6}, vui64{5} });
    REQUIRE(h.spatialNeighborsOn(2) == vvui64{ vui64{1, 2}, vui64{0, 2}, vui64{0, 1} });
    REQUIRE(h.spatialNeighborsOn(3) == vvui64{ {} });

}

void testMergeNodesSynth()
{
    utils::Hierarchy h;

    h.numComponents.emplace_back(9);
    h.numComponents.emplace_back(3);

    vui64 parentsOfData = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    h.parents.push_back(parentsOfData);

    vvui64 childrenOfMerged = { {0, 1, 2}, {3, 4, 5}, {6, 7, 8} };
    h.children.push_back(childrenOfMerged);

    std::vector<SparseVecSPH> randomWalkOnData;
    randomWalkOnData.resize(9);

    for (auto& sparseRow : randomWalkOnData)
        sparseRow.resize(9);

    randomWalkOnData[0].insert(1) = 7;
    randomWalkOnData[0].insert(2) = 8;
    randomWalkOnData[1].insert(0) = 9;
    randomWalkOnData[1].insert(2) = 11;
    randomWalkOnData[1].insert(8) = 6;
    randomWalkOnData[2].insert(0) = 4;
    randomWalkOnData[2].insert(1) = 2;
    randomWalkOnData[2].insert(3) = 3;
    randomWalkOnData[2].insert(5) = 13;
    randomWalkOnData[3].insert(2) = 1;
    randomWalkOnData[3].insert(4) = 7;
    randomWalkOnData[3].insert(5) = 5;
    randomWalkOnData[4].insert(3) = 9;
    randomWalkOnData[4].insert(6) = 19;
    randomWalkOnData[5].insert(2) = 2;
    randomWalkOnData[5].insert(3) = 6;
    randomWalkOnData[5].insert(6) = 21;
    randomWalkOnData[6].insert(4) = 5;
    randomWalkOnData[6].insert(5) = 9;
    randomWalkOnData[6].insert(7) = 3;
    randomWalkOnData[6].insert(8) = 8;
    randomWalkOnData[7].insert(6) = 5;
    randomWalkOnData[7].insert(8) = 9;
    randomWalkOnData[8].insert(1) = 4;
    randomWalkOnData[8].insert(6) = 6;
    randomWalkOnData[8].insert(7) = 2;

    h.randomWalks.push_back(randomWalkOnData);

    assert(h.numComponents.back() == childrenOfMerged.size());

    std::vector<SparseVecSPH> mergedNodes = utils::mergeNodesRandomWalks(randomWalkOnData, childrenOfMerged.size(), parentsOfData, false, false);

    /*
     Data level matrix
      0  7  8  0  0  0  0  0  0
      9  0 11  0  0  0  0  0  6
      4  2  0  3  0 13  0  0  0
      0  0  1  0  7  5  0  0  0
      0  0  0  9  0  0 19  0  0
      0  0  2  6  0  0 21  0  0
      0  0  0  0  6  9  0  3  8
      0  0  0  0  0  0  5  0  9
      0  4  0  0  0  0  6  2  0
    */
    std::cout << "Data level matrix" << "\n";
    utils::printSparseMatrixAsDense(randomWalkOnData, true, { 0, 3, 3, 3 });

    for (uint64_t i = 0; i < h.numComponents.front(); i++)
        std::cout << i << " merges into " << h.parentsOn(0)[i] << "\n";
    std::cout << "\n";

    /*
     Merged matrix
     41 16  6
      3 27 40
      4 14 33
    */
    std::cout << "Merged matrix " << "\n";
    utils::printSparseMatrixAsDense(mergedNodes, true, { 0, 3, 3, 3 });

    // Results
    std::vector<SparseVecSPH> mergeNotesResults;
    mergeNotesResults.resize(3);

    for (auto& sparseRow : mergeNotesResults)
        sparseRow.resize(3);

    mergeNotesResults[0].insert(0) = 41;
    mergeNotesResults[0].insert(1) = 16;
    mergeNotesResults[0].insert(2) = 6;
    mergeNotesResults[1].insert(0) = 3;
    mergeNotesResults[1].insert(1) = 27;
    mergeNotesResults[1].insert(2) = 40;
    mergeNotesResults[2].insert(0) = 4;
    mergeNotesResults[2].insert(1) = 14;
    mergeNotesResults[2].insert(2) = 33;

    // Check results
    REQUIRE(mergedNodes[0].isApprox(mergeNotesResults[0]));
    REQUIRE(mergedNodes[1].isApprox(mergeNotesResults[1]));
    REQUIRE(mergedNodes[2].isApprox(mergeNotesResults[2]));

}

void testMergeNodesGraph()
{
    int64_t numPoints = 9;
    int64_t k = 4;
    vui64 parentsOfData = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    vvui64 childrenOfMerged = { {0, 1, 2}, {3, 4, 5}, {6, 7, 8} };

    utils::Graph knnGraph;
    knnGraph.numPoints = numPoints;

    // pre-defined neighbors
    knnGraph.knnIndices = {
     0,  1,  2,  6,      1,  2,  8,  4,      2,  7,  4,  5, 
     3,  2,  5,  6,      4,  8,  7,  3,      5,  6,  7,  3,
     6,  4,  7,  1,      7,  6,  0,  1,      8,  0,  2,  3
    };

    knnGraph.knnDistances = {
     0.f, .1f ,  .2f,  .4f,     0.f, .2f , .3f,  .4f,     0.f,  .3f,  .4f,  .5f, 
     0.f, .8f , 1.2f, 3.f ,     0.f, .6f , .7f, 1.5f,     0.f, .10f,  .7f, 1.f , 
     0.f, .8f , 2.f , 3.f ,     0.f, .14f, .7f, 2.f ,     0.f, .13f,  .2f, 1.1f
    };

    knnGraph.updateFixedNumNeighbors(k);

    REQUIRE(knnGraph.isValid());

    SparseMatSPH matrixGraph;
    utils::convertGraphToEigenSparse(knnGraph.getGraphView(), matrixGraph);

    std::cout << "Data graph" << "\n";
    utils::printSparseMatrixAsDense(matrixGraph, true);

    SparseMatSPH mergedNodes = utils::mergeNodesDataDistances(matrixGraph, childrenOfMerged.size(), parentsOfData);
    utils::removeDiagonalElements(mergedNodes);

    utils::Graph mergedGraph = utils::mergeGraphNodes(knnGraph, childrenOfMerged.size(), parentsOfData);

    std::cout << "Merged graph" << "\n";
    utils::printSparseMatrixAsDense(mergedNodes, true);

    std::cout << "Merged graph2" << "\n";
    SparseMatSPH matrixGraphMerged;
    utils::convertGraphToEigenSparse(mergedGraph.getGraphView(), matrixGraphMerged);
    utils::printSparseMatrixAsDense(matrixGraphMerged, true);

    // Merged
    // Node 0
    // From 0, 1, 2
    // To   0, 1, 2, 4, 5, 6, 7, 8
    // New  0  0  0  1  1  2  2  2
    // distances to 1: .4f (from 1), .5f (from 2) -> should yield 0.4f
    // distances to 2: .4f (from 0), .3f (from 2), .3f (from 1) -> should yield 0.3f
    //
    // Node 1
    // From 3, 4, 5
    // To   2, 3, 4, 5, 6, 7, 8
    // New  0  1  1  1  2  2  2
    // distances to 0: .8f (from 3) -> should yield 0.8f
    // distances to 2: 3.f (from 3), .6f (from 4), .7f (from 4), .10f  (from 5), .7f (from 5) -> should yield 0.1f
    //
    // Node 2
    // From 6, 7, 8
    // To   0, 1, 2, 3, 4, 6, 7, 8
    // New  0  0  0  1  1  2  2  2
    // distances to 0: 3.f (from 6), 2.f (from 7), .7f (from 7), .13f (from 8) -> should yield 0.13f
    // distances to 1: .8f (from 6), 1.1f (from 8) -> should yield 0.8f

    REQUIRE(mergedGraph.getNeighborN(0, 1) == 2);
    REQUIRE(mergedGraph.getNeighborN(0, 2) == 1);
    CHECK_THAT(mergedGraph.getDistanceN(0, 1), Catch::Matchers::WithinRel(0.3f, eps));
    CHECK_THAT(mergedGraph.getDistanceN(0, 2), Catch::Matchers::WithinRel(0.4f, eps));

    REQUIRE(mergedGraph.getNeighborN(1, 1) == 2);
    REQUIRE(mergedGraph.getNeighborN(1, 2) == 0);
    CHECK_THAT(mergedGraph.getDistanceN(1, 1), Catch::Matchers::WithinRel(0.1f, eps));
    CHECK_THAT(mergedGraph.getDistanceN(1, 2), Catch::Matchers::WithinRel(0.8f, eps));

    REQUIRE(mergedGraph.getNeighborN(2, 1) == 0);
    REQUIRE(mergedGraph.getNeighborN(2, 2) == 1);
    CHECK_THAT(mergedGraph.getDistanceN(2, 1), Catch::Matchers::WithinRel(0.13f, eps));
    CHECK_THAT(mergedGraph.getDistanceN(2, 2), Catch::Matchers::WithinRel(0.8f, eps));

    CHECK_THAT(matrixGraphMerged[0].coeff(1), Catch::Matchers::WithinRel(0.4f, eps));
    CHECK_THAT(matrixGraphMerged[0].coeff(2), Catch::Matchers::WithinRel(0.3f, eps));
    CHECK_THAT(matrixGraphMerged[1].coeff(0), Catch::Matchers::WithinRel(0.8f, eps));
    CHECK_THAT(matrixGraphMerged[1].coeff(2), Catch::Matchers::WithinRel(0.1f, eps));
    CHECK_THAT(matrixGraphMerged[2].coeff(0), Catch::Matchers::WithinRel(0.13f, eps));
    CHECK_THAT(matrixGraphMerged[2].coeff(1), Catch::Matchers::WithinRel(0.8f, eps));

}


static void writeMatrixToCSV(const std::vector<std::vector<float>>& matrix, const std::string& filename) {
    std::ofstream file(filename);  // Open the file for writing

    if (!file.is_open()) {
        std::cerr << "Could not open the file for writing!" << std::endl;
        return;
    }

    // Optional: Set precision and format floats
    file << std::fixed << std::setprecision(6);  // Adjust the precision if necessary

    for (const auto& row : matrix) {
        for (size_t col = 0; col < row.size(); ++col) {
            file << row[col];
            if (col < row.size() - 1) {
                file << ",";  // Add comma except after the last element in a row
            }
        }
        file << "\n";  // Newline at the end of each row
    }

    file.close();  // Close the file after writing
}

static void writeVectorToCSV(const std::vector<uint64_t>& vec, const std::string& filename) {
    std::ofstream file(filename);  // Open the file for writing

    if (!file.is_open()) {
        std::cerr << "Could not open the file for writing!" << std::endl;
        return;
    }

    // Optional: Set precision and format floats
    file << std::fixed << std::setprecision(0);  // Adjust the precision if necessary

    for (size_t id = 0; id < vec.size(); ++id) {
        file << vec[id];
        if (id < vec.size() - 1) {
            file << ",";  // Add comma except after the last element in a row
        }
    }
    file << "\n";  // Newline at the end of each row

    file.close();  // Close the file after writing
}

static std::vector<std::vector<float>> convertToDense(const std::vector<SparseVecSPH>& sparseVecs)
{
    std::vector<std::vector<float>> denseMatrix;
    for (const auto& sparseVec : sparseVecs) {
        auto& denseRow = denseMatrix.emplace_back(sparseVecs.size(), 0.0f);

        for (SparseVecSPH::InnerIterator it(sparseVec); it; ++it)
            denseRow[it.index()] = it.value();
    }
    return denseMatrix;
}

void _testSaveNodeData()
{
    // Load image, assuming starting from a build in build folder in the source dir
    const auto imageDir = fs::current_path().parent_path().parent_path() / "tests" / "data" / "SquareColors10x_noisy";

    if (!std::filesystem::exists(imageDir))
    {
        std::cout << "testMergeNodesData: test data dir does not exist, skipping test..." << std::endl;
        return;
    }

    utils::ImageStack img;
    img = utils::loadTiffImageStack(imageDir);
    utils::reorderImageDataVector(img);

    // Compute kNN
    NearestNeighbors NN(img.data);
    NN.setCachingActive(false);

    // Later make this a parameter and call this function twice so test
    // once with and once without connected components.
    NearestNeighborsSettings nns;
    nns.numNearestNeighbors = 25;
    nns.knnMetric = utils::KnnMetric::L2;
    nns.knnIndex = utils::KnnIndex::BruteForce;

    NN.compute(nns);

    // There are 4 connected components in this dataset for small k and 1 for large knn (e.g. 50)
    [[maybe_unused]] std::pair<int64_t, std::shared_ptr<std::vector<int64_t>>> cc = NN.computeConnectedComponents();

    ImageHierarchy IH(NN.getKnnGraphView(), img.data, img.width, img.height);
    IH.setCachingActive(false);

    utils::RandomWalkSettings rws;
    ImageHierarchySettings ihs;
    ihs.componentSim = utils::ComponentSim::NEIGH_WALKS;
    ihs.mergeMultiple = false;
    ihs.usePercentile = false;
    ihs.maxDist = 0;
    IH.compute(ihs, rws);

    // Save random walk similarities to disk
    const auto saveDir = imageDir.parent_path() / "out";
    utils::ensurePathExists(saveDir);

    const auto& randomWalks = IH.getHierarchy().randomWalks;
    const auto& parents = IH.getHierarchy().parents;

    for (size_t level = 0; level < randomWalks.size(); level++)
    {
        std::vector<std::vector<float>> matrix = convertToDense(randomWalks[level]);
        auto savePath = saveDir / ("rws_" + std::to_string(level) + ".csv");
        writeMatrixToCSV(matrix, savePath.string());
        if (level + 1 < randomWalks.size())
        {
            savePath = saveDir / ("parents_" + std::to_string(level) + ".csv");
            writeVectorToCSV(parents[level], savePath.string());
        }
    }


}