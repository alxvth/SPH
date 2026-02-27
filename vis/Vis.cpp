#include <sph/utils/TestData.hpp>

#include <sph/utils/Logger.hpp>
#include <sph/utils/Math.hpp>

#include "Renderer.hpp"
#include "UtilsCompute.hpp"
#include "UtilsDefaults.hpp"

#include <cstdint>
#include <stdexcept>

using namespace sph;

int main()
{
	Log::info("Create test data");

	// Create test data
	int64_t numPoints = 1500;
	int64_t numDimensions = 3;

	utils::Data data;
	data.numDimensions = numDimensions;
	data.numPoints = numPoints;

	std::vector<float>& pointPos = data.getData();
	std::vector<float> pointCol;
	float noise = vis::NoiseDefaultSwissRole;

	utils::createSwissRole(pointPos, pointCol, numPoints, noise);

	// normalize to [-1, 1]
	utils::normalizeScale(pointPos.begin(), pointPos.end());

	// create line for the shortest path
	std::vector<float> knnLinePos, knnLineCol, knnDists;
	std::vector<int64_t> knnIdx;
	int64_t k = 10;

	{
		auto knnGraph = vis::computeKnn(data, k);
		vis::appendKnnLines(data, knnGraph.knnIndices, k, knnLinePos, knnLineCol);
	}

	Log::info("Initialize renderer");
	Renderer renderer(std::move(pointPos), std::move(pointCol), std::move(knnLinePos), std::move(knnLineCol));
	try
	{
		renderer.initWindow();
	}
	catch (const std::runtime_error& e)
	{
		fmt::println("{}", e.what());
	}

	Log::info("Init opengl buffers");
	renderer.initBuffers();

	if (renderer.isInit())
	{
		Log::info("Start rendering");
		renderer.render();
	}

	Log::info("End.");

	return 0;
}

