#pragma once

#include <cstdint>
#include <vector>

namespace sph::utils
{
	struct Data;
}

void printPath(const std::vector<int64_t>& path, float dist);

float distanceL2(const std::vector<float>& data, int64_t numDimensions, int64_t startID, int64_t endID);

void testAStar(const std::vector<int64_t>& numNeighbors, int64_t numRuns, const sph::utils::Data& data, const int verbosity = 0);

void testShortestPathCaching(const std::vector<int64_t>& numNeighbors, int64_t numRuns, const sph::utils::Data& data, const int verbosity = 0);
