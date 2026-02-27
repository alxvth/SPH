#pragma once

#include <cstdint>
#include <vector>

namespace sph::utils
{
	struct DataView;
	struct Graph;
	struct GraphView;
}

namespace Eigen
{
	template <typename Scalar, int Options, typename Index>
	class SparseVector;
}

namespace tinycolormap
{
	enum class ColormapType;
}

class Renderer;

namespace vis
{
	sph::utils::Graph computeKnn(const sph::utils::DataView& data, uint64_t k);

	void computeShortestPath(const sph::utils::DataView& data, const sph::utils::GraphView& knnGraph, int64_t startID, int64_t endID, std::vector<int64_t>& geodesic);

	void appendGeodesicLines(const sph::utils::DataView& data, const std::vector<int64_t>& geodesic, const std::array<float, 3>& lineColRGB, std::vector<float>& linePos, std::vector<float>& lineCol);

	void appendKnnLines(const sph::utils::DataView& data, const std::vector<int64_t>& knn_idx, int64_t k, std::vector<float>& linePos, std::vector<float>& lineCol);

	void shortestPathNodes(const sph::utils::DataView& data, const sph::utils::GraphView& knnGraph, const std::array<float, 3>& lineColRGB, std::vector<float>& lineIDs, std::vector<float>& lineCol, int64_t& startID, int64_t& endID, bool random = true);
	
	void doRandomWalk(const sph::utils::Graph& knnGraph, int numWalks, int walkLength, int gaussNormDist, int stepWeight, std::vector<Eigen::SparseVector<float, 0, int>>& similarities, std::vector<Eigen::SparseVector<float, 0, int>>& randomWalks, float& maxVal);

	void assignRandomWalkColors(Renderer* renderer, const tinycolormap::ColormapType& tnycolormap, float randomWalkMaxVal, const int pointID);

	void assignKnnLineWidth(int weightingType, float constWidth, const std::vector<Eigen::SparseVector<float, 0, int>>& similarities, const std::vector<float>& distances, const std::vector<int64_t>& distancesIDx, float widthMultiplier, int selectedPoint, int64_t k, std::vector<float>& knnLinesWidth);
}
