#include "UtilsCompute.hpp"

#include "Renderer.hpp"

#include <sph/NearestNeighbors.hpp>
#include <sph/utils/AStar.hpp>
#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/GraphNormalization.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/SparseMatrixAlgorithms.hpp>
#include <sph/utils/Statistics.hpp>
#include <sph/utils/Timer.hpp>

#include <algorithm>
#include <cassert>
#include <random>

#include <Eigen/SparseCore>
#include <tinycolormap.hpp>

using namespace sph;

namespace vis
{

	utils::Graph computeKnn(const utils::DataView& data, uint64_t k)
	{
		auto knn = NearestNeighbors(data);
		knn.setCachingActive(false);
		NearestNeighborsSettings nns;
		nns.numNearestNeighbors = k;
		nns.knnMetric = utils::KnnMetric::L2;
		nns.knnIndex = utils::KnnIndex::Flat;	// Alternative for larger data -> KnnIndex::IVFFlat
		knn.compute(nns);

		return knn.getKnnGraph();
	}

	void computeShortestPath(const utils::DataView& data, const utils::GraphView& knnGraph, int64_t startID, int64_t endID, std::vector<int64_t>& geodesic) {

		float g_dist(0);
		{
			utils::ScopedTimer<> AStarTimer("Compute shortest path (A*)");
			utils::astar(knnGraph, data, startID, endID, geodesic, g_dist, /*verbose*/ 2);
		}

		Log::info("Geodesic between nodes: ({0}, {1})", startID, endID);
		Log::info("Euclid. dist : {}", utils::astarDistanceHeuristic(data, startID, endID));
		Log::info("Geodesic dist: {}", g_dist);
	}

	void appendGeodesicLines(const utils::DataView& data, const std::vector<int64_t>& geodesic, const std::array<float, 3>& lineColRGB, std::vector<float>& linePos, std::vector<float>& lineCol)
	{
		linePos.clear();
		lineCol.clear();

		int64_t current(0), next(0);

		auto numLines = geodesic.size() - 1;

		linePos.reserve(numLines * 6);

		// append geodesic lines
		for (size_t i = 0; i < numLines; i++)
		{
			current = geodesic[i];
			next = geodesic[i + 1ll];

			// Point positions
			linePos.emplace_back(data[current * 3 + 0]);
			linePos.emplace_back(data[current * 3 + 1]);
			linePos.emplace_back(data[current * 3 + 2]);

			linePos.emplace_back(data[next * 3 + 0]);
			linePos.emplace_back(data[next * 3 + 1]);
			linePos.emplace_back(data[next * 3 + 2]);
		}

		// append colors
		lineCol.resize(linePos.size());

		for (size_t i = 0; i < linePos.size(); i += 3) {
			lineCol[i] = lineColRGB[0];
			lineCol[i + 1] = lineColRGB[1];
			lineCol[i + 2] = lineColRGB[2];
		}

	}

	void appendKnnLines(const utils::DataView& data, const std::vector<int64_t>& knn_idx, int64_t k, std::vector<float>& linePos, std::vector<float>& lineCol)
	{
		linePos.clear();
		lineCol.clear();

		const float color = 0.5f;
		int64_t next(0);

		int64_t numPoints = knn_idx.size() / k;

		linePos.reserve(numPoints * 6 * (k - 1));

		// append knn lines
		for (int64_t current = 0; current < numPoints; current++)
		{
			for (int64_t j = 1; j < k; j++)
			{
				next = knn_idx[current * k + j];

				linePos.emplace_back(data[current * 3 + 0]);
				linePos.emplace_back(data[current * 3 + 1]);
				linePos.emplace_back(data[current * 3 + 2]);

				linePos.emplace_back(data[next * 3 + 0]);
				linePos.emplace_back(data[next * 3 + 1]);
				linePos.emplace_back(data[next * 3 + 2]);
			}
		}

		// append colors
		lineCol.resize(linePos.size(), color);

	}

	void shortestPathNodes(const utils::DataView& data, const utils::GraphView& knnGraph, const std::array<float, 3>& lineColRGB, std::vector<float>& lineIDs, std::vector<float>& lineCol, int64_t& startID, int64_t& endID, bool random)
	{
		if (random)
		{
			std::random_device rd;  // a seed source for the random number engine
			std::mt19937_64 gen(rd()); // mersenne_twister_engine seeded with rd()
			std::uniform_int_distribution<unsigned int> distrib(0, static_cast<unsigned int>(data.getNumPoints()));

			startID = distrib(gen);
			endID = distrib(gen);
		}

		if (startID == endID)
		{
			lineIDs.clear();
			lineCol.clear();
			return;
		}

		std::vector<int64_t> geodesic;
		computeShortestPath(data, knnGraph, startID, endID, geodesic);
		appendGeodesicLines(data, geodesic, lineColRGB, lineIDs, lineCol);
	}

	void doRandomWalk(const utils::Graph& knnGraph, int numWalks, int walkLength, int gaussNormDist, int stepWeight, std::vector<Eigen::SparseVector<float, 0, int>>& similarities, std::vector<Eigen::SparseVector<float, 0, int>>& randomWalks, float& maxVal)
	{
		const auto graphView = knnGraph.getGraphView();
		switch (gaussNormDist)
		{
		case 0:  utils::computeLinearDistributions(graphView, similarities);				break;
		case 1:  utils::computeGaussianDistributions(graphView, similarities);			break;
		}

		for (auto& similarities : similarities)
			utils::normalizeSparseVector(similarities);

		utils::RandomWalkSettings randomWalkSettings;
		randomWalkSettings.numRandomWalks = static_cast<uint64_t>(numWalks);
		randomWalkSettings.singleWalkLength = static_cast<uint64_t>(walkLength);

		switch (stepWeight)
		{
		case 0:  randomWalkSettings.importanceWeighting = utils::ImportanceWeighting::CONSTANT; break;
		case 1:  randomWalkSettings.importanceWeighting = utils::ImportanceWeighting::LINEAR;   break;
		case 2:  randomWalkSettings.importanceWeighting = utils::ImportanceWeighting::NORMAL;   break;
		case 3:  randomWalkSettings.importanceWeighting = utils::ImportanceWeighting::ONLYLAST;
		}

		utils::SparseMatrixStats randomWalkStats;
		utils::doRandomWalks(similarities, randomWalkSettings, randomWalks, randomWalkStats);
		utils::removeDiagonalElements(randomWalks);

		for (auto& randomWalk : randomWalks)
			//utils::normalizeMinMaxSparseVector(randomWalk);
			utils::normalizeSparseVector(randomWalk);

		// get global max entry
		maxVal = 0;
		float currentMax = 0;
		for (auto& randomWalk : randomWalks)
		{
			currentMax = randomWalk.coeffs().maxCoeff();
			if (currentMax >= maxVal)
				maxVal = currentMax;
		}

	}

	void assignRandomWalkColors(Renderer* renderer, const tinycolormap::ColormapType& tnycolormap, float randomWalkMaxVal, const int pointID)
	{
		const auto& randomWalks = renderer->getRandomWalks();
		auto& pointsCol = renderer->getPointColors();

		// set all point colors to default
		tinycolormap::Color color = tinycolormap::GetColor(0, tnycolormap);

		SPH_PARALLEL
		for (size_t i = 0; i < pointsCol.size(); i += 3) {
			std::transform(color.data, color.data + 3, pointsCol.begin() + i, [](double value) {
				return static_cast<float>(value);
				});
		}

		// assign color values for non-zero walk entries
		auto& currentWalks = randomWalks[pointID];
		for (Eigen::SparseVector<float>::InnerIterator it(currentWalks); it; ++it)
		{
			color = tinycolormap::GetColor(std::sqrt(it.value()) / std::sqrt(randomWalkMaxVal), tnycolormap);
			pointsCol[it.index() * 3ll    ] = static_cast<float>(color.r());
			pointsCol[it.index() * 3ll + 1] = static_cast<float>(color.g());
			pointsCol[it.index() * 3ll + 2] = static_cast<float>(color.b());
		}

		pointsCol[pointID * 3ll    ] = 1;
		pointsCol[pointID * 3ll + 1] = 1;
		pointsCol[pointID * 3ll + 2] = 1;

		Log::info("Random walk values for {}: ", pointID);
		utils::printSparseVector(currentWalks, true);

		renderer->rebindPointData();
	}


	void assignKnnLineWidth(int weightingType, float constWidth, const std::vector<Eigen::SparseVector<float, 0, int>>& similarities, const std::vector<float>& distances, const std::vector<int64_t>& distancesIDx, float widthMultiplier, int selectedPoint, int64_t k, std::vector<float>& knnLinesWidth)
	{
		switch (weightingType) {
			case 0: // equal weights
				assert(constWidth >= 0);
				std::fill(knnLinesWidth.begin(), knnLinesWidth.end(), constWidth);
				break;
			case 1: // distances
			{
				assert(selectedPoint >= 0);
				assert(k >= 1);

				std::fill(knnLinesWidth.begin(), knnLinesWidth.end(), 0.f);
				
				float dist = 0;
				int64_t id = selectedPoint * (k - 1) * 2;
				std::cout << "distances: " << k - 1 << "\n";

				for (int64_t n = 0; n < (k - 1); n++)
				{
					assert(distances[selectedPoint * k + n + 1] >= 0);
					dist = widthMultiplier * 2 * (1 - distances[selectedPoint * k + n + 1]);
					knnLinesWidth[id + 2 * n]		= dist;
					knnLinesWidth[id + 2 * n + 1]	= dist;

					std::cout << "To " << distancesIDx[selectedPoint * k + n + 1] << ": " << distances[selectedPoint * k + n + 1] << "\n";
				}
				std::cout << std::endl;
				break;
			}
			case 2: // similarities
			{
				assert(selectedPoint >= 0);
				assert(k >= 1);

				if (similarities.empty())
				{
					Log::warn("Enable 'Compute random walk on click' to resize line with similarites");
					break;
				}

				std::fill(knnLinesWidth.begin(), knnLinesWidth.end(), 0.f);

				float sim = 0;
				int64_t id = selectedPoint * (k - 1) * 2;
				std::cout << "similarities: " << similarities[selectedPoint].nonZeros() << "\n";

				const auto& sparseSims = similarities[selectedPoint];
				std::vector<float> simVec;
				simVec.reserve(sparseSims.nonZeros());

				for (Eigen::SparseVector<float>::InnerIterator it(sparseSims); it; ++it)
					simVec.push_back(it.value());

				std::sort(simVec.begin(), simVec.end());

				for (int64_t n = 0; n < (k - 1); n++)
				{
					assert(simVec[n] >= 0);
					sim = widthMultiplier * (1 - simVec[n]);
					knnLinesWidth[id + 2 * n]		= sim;
					knnLinesWidth[id + 2 * n + 1]	= sim;
					std::cout << id + 2 * n << " & " << id + 2 * n + 1 << ": " << simVec[n] << "\n";
				}
				std::cout << std::endl;
				break;
			}
		}


	}

} // namespace vis
