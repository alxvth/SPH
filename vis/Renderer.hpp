#pragma once

#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <sph/utils/Data.hpp>
#include <sph/utils/Graph.hpp>

#include <glm/vec3.hpp>

#include <Eigen/SparseCore>

#include "Shader.hpp"

struct GLFWwindow;

using Color = glm::vec3;

class Renderer
{
public:
	Renderer(std::vector<float>&& pointPos, std::vector<float>&& pointCol, std::vector<float>&& knnLinePos, std::vector<float>&& knnLineCol);
	~Renderer();

	void render();

	void initWindow();
	void initBuffers();

	bool isInit() const { return _init; }

	int getWindowHeight() const { return _windowHeight; }
	void setWindowHeight(int newHeight) { 
		_windowHeight = newHeight; 
	}

	int getWindowWidth() const { return _windowWidth; }
	void setWindowWidth(int newWidth) { 
		_windowWidth = newWidth; 
	}

	const std::vector<Eigen::SparseVector<float>>& getSimilarities() const { return _similarities; }
	const std::vector<Eigen::SparseVector<float>>& getRandomWalks() const { return _randomWalks; }

	const std::vector<float>& getKnnDists() const { return _knnGraph.knnDistances; }
	const std::vector<int64_t>& getKnnIdx() const { return _knnGraph.knnIndices; }

	const std::vector<float>& getKnnLinesWidth() const { return _knnLinesWidth; };
	std::vector<float>& getKnnLinesWidth() { return _knnLinesWidth; };

	int64_t getKnNumber() const { return _knnGraph.getK(); };

	const std::vector<float>& getPointPositions() const { return _pointsPos; };
	std::vector<float>& getPointPositions() { return _pointsPos; };

	const std::vector<float>& getPointColors() const { return *_pointsCol; };
	std::vector<float>& getPointColors() { return *_pointsCol; };

	void useDataPointColor() { 
		_pointsCol = &_pointsColData; 
	}

	void useRandomWalkPointColor() { 
		_pointsCol = &_pointsColRandom; 
	}

	void setColor(size_t ID, float r, float g, float b) { 
		(*_pointsCol)[ID * 3]	  = r; 
		(*_pointsCol)[ID * 3 + 1] = g;
		(*_pointsCol)[ID * 3 + 2] = b;
	};

	void rebindPointData() const;

private:
	void drawLine(int64_t nodeA, int64_t nodeB, Color colA = Color(0.1f, 0.1f, 0.1f), Color colB = Color(0.1f, 0.1f, 0.1f), float lineWidth = 1.f, float opacity = 0.9f);
	void drawLines(std::vector<float>& linePos, std::vector<float>& lineCol, std::vector<float>& lineWidths, float opacity = 0.9f) const ;
	void drawPoints(float opacity = 0.9f, bool rebind = false) const;

private:
	bool _init = false;
	GLFWwindow* _window = nullptr;
	int _windowHeight = 0;
	int _windowWidth = 0;

	unsigned int _vaPoints = 0;
	unsigned int _vaLines = 0;
	unsigned int _vaLineSingle = 0;

	unsigned int _vbPointsPos = 0;
	unsigned int _vbPointsCol = 0;
	unsigned int _vbPointsSize = 0;
	unsigned int _vbLinesPos = 0;
	unsigned int _vbLinesCol = 0;
	unsigned int _vbLinesWidth = 0;
	unsigned int _vbLineSinglePos = 0;
	unsigned int _vbLineSingleCol = 0;
	unsigned int _vbLineSingleWidth = 0;

	const unsigned int _posAttribute = 0;
	const unsigned int _colAttribute = 1;
	const unsigned int _widthAttribute = 2;

	sph::utils::DataView _data = {};
	std::vector<float> _pointsPos = {};
	std::vector<float> _pointsColData = {};
	std::vector<float> _pointsColRandom = {};
	std::vector<float>* _pointsCol = nullptr;
	std::vector<float> _pointsSize = {};
	std::vector<float> _geodesicLinesPos = {};
	std::vector<float> _geodesicLinesCol = {};
	std::vector<float> _geodesicLinesWidth = {};
	std::array<float, 6> _lineSinglePos = {};
	std::array<float, 6> _lineSingleCol = {};
	std::array<float, 2> _lineSingleWidth = {};

	std::unique_ptr<Shader> _pointShader = {};
	std::unique_ptr<Shader> _lineShader = {};

	sph::utils::Graph _knnGraph = {};

	std::vector<float>   _knnLinesPos = {};
	std::vector<float>   _knnLinesCol = {};
	std::vector<float>	 _knnLinesWidth = {};

	std::vector<Eigen::SparseVector<float>> _similarities = {};
	std::vector<Eigen::SparseVector<float>> _randomWalks = {};
};
