#include "AStarTest.hpp"

#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sph/NearestNeighbors.hpp>

#include "sph/utils/AStar.hpp"
#include "sph/utils/AStarBoost.hpp"
#include <sph/utils/Data.hpp>
#include <sph/utils/Distances.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/ShortestPath.hpp>
#include <sph/utils/Timer.hpp>

#include <catch2/catch_test_macros.hpp>	
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <fmt/format.h>

using namespace sph;

constexpr float eps = 0.0001f;

void testAStar(const std::vector<int64_t>& numNeighbors, int64_t numRuns, const utils::Data& data, const int verbosity)
{
    // Helper to create random nodes
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned int> distrib = std::uniform_int_distribution<unsigned int>(0, static_cast<unsigned int>(data.getNumPoints() - 1));

    int64_t startID(-1), endID(-1);

    // Results
    std::vector<int64_t> geodesic, BOOST_geodesic, BOOST_SPH_geodesic;
    float g_dist(-1.f), BOOST_g_dist(-1.f), BOOST_SPH_g_dist(-1.f);

    // Setup Knn
    auto knn = NearestNeighbors(data);
    knn.setCachingActive(false);

    NearestNeighborsSettings nns;
    nns.knnMetric = utils::KnnMetric::L2;
    nns.knnIndex = utils::KnnIndex::Flat;   // Alternative for larger data -> KnnIndex::IVFFlat
    
    if (data.getNumPoints() > 100000)
        nns.knnIndex = utils::KnnIndex::IVFFlat;

    // For several k nearest neighbor graphs compute multiple shortest paths
    for (const auto k : numNeighbors)
    {
        if (verbosity >= 0)
            Log::info("Compute knn graph with " + std::to_string(k) + " nearest neighbors");

        nns.numNearestNeighbors = k;

        knn.compute(nns);

        const auto& knnGraph = knn.getKnnGraph();
        const auto& knnGraphView = knnGraph.getGraphView();

        auto createBoostGraph_start = utils::clock::now();
        utils::BoostGraph bgraph = utils::createBoostGraph(knnGraph);
        auto createBoostGraph_end = utils::clock::now();
        auto createBoostGraph_len = std::chrono::duration_cast<std::chrono::milliseconds>(createBoostGraph_end - createBoostGraph_start);
        Log::info("Create Boost Graph: {} ms", createBoostGraph_len.count());

        std::chrono::microseconds total_duration_sph{0};
        std::chrono::microseconds total_duration_boost{0};
        std::chrono::microseconds total_duration_boost_sph{0};

        for (int64_t run = 0; run < numRuns; run++)
        {
            startID = distrib(gen);
            endID = distrib(gen);

            Log::info(std::format("## Path {0}/{1} between nodes {2} and {3} ({4} nn) ##", run + 1, numRuns, startID, endID, k));

            // This library
            {
                auto sph_start = utils::clock::now();
                utils::astar(knnGraph, data, startID, endID, geodesic, g_dist);
                auto sph_end = utils::clock::now();

                total_duration_sph += std::chrono::duration_cast<std::chrono::microseconds>(sph_end - sph_start);
            }

            // Reference implementation
            {
                std::vector<float> distance_map;
                std::vector<uint64_t> predecessor_map;
                auto boost_start = utils::clock::now();
                utils::astarBoost(bgraph, data, startID, endID, distance_map, predecessor_map, BOOST_geodesic, BOOST_g_dist);
                auto boost_end = utils::clock::now();

                total_duration_boost += std::chrono::duration_cast<std::chrono::microseconds>(boost_end - boost_start);
            }

            // Reference implementation
            {
                std::vector<float> distance_map_sph;
                std::vector<uint64_t> predecessor_map_sph;
                auto boost_sph_start = utils::clock::now();
                utils::astarBoost(knnGraphView, data, startID, endID, distance_map_sph, predecessor_map_sph, BOOST_SPH_geodesic, BOOST_SPH_g_dist);
                auto boost_sph_end = utils::clock::now();

                total_duration_boost_sph += std::chrono::duration_cast<std::chrono::microseconds>(boost_sph_end - boost_sph_start);
            }

            // Check validity
            if (g_dist < 0)
                Log::warn("SPH graph did not find a shortest path");

            if (BOOST_g_dist < 0)
                Log::warn("Boost graph did not find a shortest path");

            if (BOOST_SPH_g_dist < 0)
                Log::warn("Boost graph did not find a shortest path");

            if (g_dist < 0 || BOOST_g_dist < 0 || BOOST_SPH_g_dist < 0)
                continue;

            // Print results
            if (verbosity >= 1)
            {
                Log::info("SPH");
                printPath(geodesic, g_dist);
                Log::info("Boost");
                printPath(BOOST_geodesic, BOOST_g_dist);
                Log::info("Boost SPH");
                printPath(BOOST_SPH_geodesic, BOOST_SPH_g_dist);
            }

            if (verbosity >= 2)
            {
                auto printPathVerbose = [&data](std::vector<int64_t>& geo, const std::string& name) {
                    Log::info("Accumulated distances on path: " + name);
                    float dist_accum(0), dist_local(0);
                    for (uint64_t i = 0; i < geo.size() - 1; i++)
                    {
                        dist_local = distanceL2(data.getData(), data.getNumDimensions(), geo[i], geo[i + 1]);
                        dist_accum += dist_local;
                        Log::info(fmt::format("{:<8} -> {:<8}: {:.5g} (+ {:.5g})", geo[i], geo[i + 1], dist_accum, dist_local));

                    }
                };

                printPathVerbose(geodesic, "SPH");
                printPathVerbose(BOOST_geodesic, "BOOST_geodesic");
                printPathVerbose(BOOST_SPH_geodesic, "BOOST_SPH_geodesic");
            }

            // Compare results
            
            // geodesic distance must be at least the euclidean distance, up to some numerical precision
            CHECK(g_dist - distanceL2(data.getData(), data.getNumDimensions(), startID, endID) >= -eps);

            // distance must equal, up to some numerical precision
            CHECK_THAT(g_dist, Catch::Matchers::WithinRel(BOOST_g_dist, eps));
            CHECK_THAT(g_dist, Catch::Matchers::WithinRel(BOOST_SPH_g_dist, eps));

            // path should be equal
            // but in some cases, when two path have the same distance, boost and sph might visit different ones
            //CHECK(geodesic == BOOST_geodesic);
        }

        Log::info("Avg SPH:         {} us", total_duration_sph.count() / numRuns);
        Log::info("Avg Boost:       {} us", total_duration_boost.count() / numRuns);
        Log::info("Avg Boost (sph): {} us", total_duration_boost_sph.count() / numRuns);


    }

}

void testShortestPathCaching(const std::vector<int64_t>& numNeighbors, int64_t numRuns, const utils::Data& data, const int verbosity)
{
    // Helper to create random nodes
    std::random_device rd;
    std::mt19937 gen = std::mt19937(rd());
    std::uniform_int_distribution<unsigned int> distrib = std::uniform_int_distribution<unsigned int>(0, static_cast<unsigned int>(data.getNumPoints() - 1));

    int64_t startID(-1), endID(-1);

    // Results
    std::vector<int64_t> BOOST_geodesic;
    float g_dist(-1.f), g_dist_lookup(-1.f), BOOST_g_dist(-1.f);

    // Setup Knn
    auto knn = NearestNeighbors(data);
    knn.setCachingActive(false);

    NearestNeighborsSettings nns;
    nns.knnMetric = utils::KnnMetric::L2;
    nns.knnIndex = utils::KnnIndex::Flat;   // Alternative for larger data -> KnnIndex::IVFFlat

    if (data.getNumPoints() > 100000)
        nns.knnIndex = utils::KnnIndex::IVFFlat;

    utils::cache::setUseCacheShortestPath(true);

    // For several k nearest neighbor graphs compute multiple shortest paths
    for (const auto k : numNeighbors)
    {
        if (verbosity >= 0)
            Log::info("Compute knn graph with " + std::to_string(k) + " nearest neighbors");

        nns.numNearestNeighbors = k;

        knn.compute(nns);

        const auto& knnGraph = knn.getKnnGraph();

        utils::BoostGraph bgraph = utils::createBoostGraph(knnGraph);

        std::chrono::microseconds total_duration_sph{ 0 };
        std::chrono::microseconds total_duration_sph_lookup{ 0 };
        std::chrono::microseconds total_duration_boost{ 0 };

        for (int64_t run = 0; run < numRuns; run++)
        {
            startID = distrib(gen);
            endID = distrib(gen);

            Log::info(std::format("## Path {0}/{1} between nodes {2} and {3} ({4} nn) ##", run + 1, numRuns, startID, endID, k));

            // This library
            auto sph_start = utils::clock::now();
            g_dist = utils::computeShortestPath(knnGraph, data, startID, endID, nullptr, std::nullopt);
            auto sph_end = utils::clock::now();

            total_duration_sph += std::chrono::duration_cast<std::chrono::microseconds>(sph_end - sph_start);

            auto sph_start_lookup = utils::clock::now();
            g_dist_lookup = utils::computeShortestPath(knnGraph, data, startID, endID, nullptr, std::nullopt);
            auto sph_end_lookup = utils::clock::now();

            total_duration_sph_lookup += std::chrono::duration_cast<std::chrono::microseconds>(sph_end_lookup - sph_start_lookup);

            // Reference implementation
            std::vector<float> distance_map;
            std::vector<uint64_t> predecessor_map;
            auto boost_start = utils::clock::now();
            utils::astarBoost(bgraph, data, startID, endID, distance_map, predecessor_map, BOOST_geodesic, BOOST_g_dist);
            auto boost_end = utils::clock::now();

            total_duration_boost += std::chrono::duration_cast<std::chrono::microseconds>(boost_end - boost_start);

            // Check validity
            if (g_dist < 0)
                Log::warn("SPH graph did not find a shortest path");

            if (g_dist_lookup < 0)
                Log::warn("SPH graph did not find a shortest path");

            if (BOOST_g_dist < 0)
                Log::warn("Boost graph did not find a shortest path");

            if (g_dist < 0 || g_dist_lookup < 0 || BOOST_g_dist < 0)
                continue;

            // Compare results

            // geodesic distance must be at least the euclidean distance, up to some numerical precision
            CHECK(g_dist - distanceL2(data.getData(), data.getNumDimensions(), startID, endID) >= -eps);

            // lookup must reproduce computation
            CHECK(g_dist == g_dist_lookup);

            // distance must equal, up to some numerical precision
            CHECK_THAT(g_dist, Catch::Matchers::WithinRel(BOOST_g_dist, eps));

        }

        Log::info("Avg SPH:   {} us", total_duration_sph.count() / numRuns);
        Log::info("Avg Boost: {} us", total_duration_boost.count() / numRuns);
        Log::info("Avg Cache: {} us", total_duration_sph_lookup.count() / numRuns);

        utils::cache::clearCacheShortestPath();
        utils::cache::setUseCacheShortestPath(false);

    }
}

void printPath(const std::vector<int64_t>& path, float dist)
{
    Log::info(fmt::format("Length: {} (Sum of squared Eucl.)", dist));
    std::string prt = "Path: ";
    for (const auto& v : path) {
        prt += std::to_string(v) + " ";
    }
    Log::info(prt);
}

float distanceL2(const std::vector<float>& data, int64_t numDimensions, int64_t startID, int64_t endID)
{
    return utils::L2(data.data() + startID * numDimensions, data.data() + endID * numDimensions, numDimensions);
}