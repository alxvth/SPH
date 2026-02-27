#include "Renderer.hpp"

#include "UtilsCompute.hpp"
#include "UtilsDefaults.hpp"

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/Math.hpp>
#include <sph/utils/TestData.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tinycolormap.hpp>

#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace sph;

// callbacks
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_click_callback(GLFWwindow* window, int button, int action, int mods);
void key_callbacks(GLFWwindow* window, int key, int scancode, int action, int mods);
void moveCameraPos(GLFWwindow* window);
void glfw_error_callback(int error, const char* description);

// camera
glm::vec3 cameraPos		= glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront	= glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp		= glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 cameraLeft	= glm::vec3(-1.0f, 0.0f, 0.0f);

glm::mat4 projectionMatrix {};
glm::mat4 viewMatrix {};

bool firstMouse		= true;
bool mouseActive	= false;
float yaw			= -90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitch			= 0.0f;
float lastX			= 800.0f / 2.0;
float lastY			= 600.0 / 2.0;
const float fov		= 45.0f;
const float nearClip= 0.1f;
const float farClip	= 100.0f;
float cameraSpeed	= 0.06f;

// rendering
std::array<float, 3> backgroundRGB = { 0.2f, 0.3f, 0.3f, };
std::array<float, 3> geodesicLineColRGB = { 0.9f, 0.9f, 0.9f, };
std::array<float, 3> euclideanLineColRGB = { 0.5f, 0.0f, 0.0f, };
float pointSize = 5.f;
float pointOpacity = 0.9f;
float singleLineWidth = 5.f;
float geodesicLinesWidth = 5.f;
float knnLinesWidth = 1.f;
size_t currentColormap = 0;
ImPlotColormap impltcolormap = vis::colormaps[currentColormap].first;	// Viridis
tinycolormap::ColormapType tnycolormap = vis::colormaps[currentColormap].second;	// Viridis

// data
int64_t numDims				= 3;
int dataSize				= 1500;
int64_t dataViewSize		= 1500;

// clicking/selecting
glm::vec3 prevSelColor		= {};
int selectedDataID			= -1;
float sphere_radius			= .005f;
float sphere_radius_squared = sphere_radius * sphere_radius;
bool drawRandomWalks		= false;
int knnLineWeighting		= 0;
bool isMenuHovered			= false;

// computations
int randomWalkNum		= 15;
int randomWalkLength	= 90;
int walkStepWeight		= 0;
int gaussianNormDist	= 1;
float randomWalkMaxVal	= 1.f;

// timing
double deltaTime	= 0.0;
double lastFrame	= 0.0;
double waitTimeout	= 1. / 20.0;
bool idleRender		= false;

Renderer::Renderer(std::vector<float>&& pointPos, std::vector<float>&& pointCol, std::vector<float>&& knnLinePos, std::vector<float>&& knnLineCol)
{
	_pointsPos = std::move(pointPos);
	_pointsColData = std::move(pointCol);
	_pointsColRandom.resize(_pointsColData.size());
	_pointsCol = &_pointsColData;
	_pointsSize.resize(_pointsPos.size() / 3, pointSize);


	_knnLinesPos = std::move(knnLinePos);
	_knnLinesCol = std::move(knnLineCol);
	_knnLinesWidth.resize(_geodesicLinesPos.size() / 3);

	_data = utils::DataView(&_pointsPos, &dataViewSize, &numDims);

	assert(_pointsPos.size() % 3 == 0);
	assert(_pointsPos.size() == _pointsColData.size());
	assert(_knnLinesPos.size() % 3 == 0);
	assert(_knnLinesPos.size() == _knnLinesCol.size());
}

Renderer::~Renderer()
{
	ImPlot::DestroyContext();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glDeleteVertexArrays(1, &_vaPoints);
	glDeleteVertexArrays(1, &_vaLines);
	glDeleteVertexArrays(1, &_vaLineSingle);
	glDeleteBuffers(1, &_vbPointsPos);
	glDeleteBuffers(1, &_vbPointsCol);
	glDeleteBuffers(1, &_vbPointsSize);
	glDeleteBuffers(1, &_vbLinesPos);
	glDeleteBuffers(1, &_vbLinesCol);
	glDeleteBuffers(1, &_vbLinesWidth);
	glDeleteBuffers(1, &_vbLineSinglePos);
	glDeleteBuffers(1, &_vbLineSingleCol);
	glDeleteBuffers(1, &_vbLineSingleWidth);

	glfwDestroyWindow(_window);
	glfwTerminate();
}

void Renderer::initWindow()
{
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		throw std::runtime_error("glfwInit failed");

	// Set GL+GLSL versions
	const char* glsl_version = "#version 330";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 3);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	_windowHeight = 720;
	_windowWidth = 1280;
	_window = glfwCreateWindow(_windowWidth, _windowHeight, "Spatial Hierarchy Visualization", nullptr, nullptr);
	if (_window == NULL)
	{
		Log::error("Failed to create GLFW window");
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(_window);

	// let callbacks access class data
	glfwSetWindowUserPointer(_window, reinterpret_cast<void*>(this));

	// glfw callbacks
	glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
	glfwSetKeyCallback(_window, key_callbacks);
	glfwSetCursorPosCallback(_window, mouse_move_callback);
	glfwSetMouseButtonCallback(_window, mouse_click_callback);

	glfwSwapInterval(1); // Enable vsync

	// load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		glfwTerminate();
		throw std::runtime_error("Failed to initialize GLAD");
	}

	// configure global opengl state
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_MULTISAMPLE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	ImGui::StyleColorsDark();

	ImPlot::CreateContext();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(_window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Shader
	std::filesystem::path current = std::filesystem::current_path();
	std::filesystem::path relativeResFolder = {};
	int goUpDirs = 0;

	auto containsFolderName = [](const std::filesystem::path& filepath, const std::string& folderName) -> bool {
		for (const auto& component : filepath) {
			if (component.string() == folderName) {
				return true;
			}
		}
		return false;
		};

	if (containsFolderName(current, "third_party"))
	{
		goUpDirs = 4;
		relativeResFolder = std::filesystem::path("third_party") / "SpatialHierarchyLibrary"/ "vis" / "res";
	}
	else if(containsFolderName(current, "vis"))
	{
		goUpDirs = 2;
		relativeResFolder = std::filesystem::path("vis") / "res";
	}

	for (int i = 0; i < goUpDirs; ++i)
		current = current.parent_path();

	auto resDir = current / relativeResFolder;
	auto vertPointShader = (resDir / "Point.vert").string();
	auto fragPointShader = (resDir / "Point.frag").string();
	auto vertLineShader = (resDir / "Lines.vert").string();
	auto geomLineShader = (resDir / "Lines.geom").string();

	_pointShader = std::make_unique<Shader>();
	_pointShader->init(vertPointShader, fragPointShader);

	_lineShader = std::make_unique<Shader>();
	_lineShader->init(vertLineShader, fragPointShader, geomLineShader);

	_init = true;
}

void Renderer::initBuffers()
{
	// /////////////// //
	// point rendering //
	// /////////////// //
	// create buffers
	glGenVertexArrays(1, &_vaPoints);
	glGenBuffers(1, &_vbPointsPos);
	glGenBuffers(1, &_vbPointsCol);
	glGenBuffers(1, &_vbPointsSize);

	glBindVertexArray(_vaPoints);

	// bind buffer and allocate point data 
	glBindBuffer(GL_ARRAY_BUFFER, _vbPointsPos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _pointsPos.size(), _pointsPos.data(), GL_STATIC_DRAW);

	// enable point attribute shader input
	glVertexAttribPointer(_posAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(_posAttribute);

	// bind buffer and allocate point color data 
	glBindBuffer(GL_ARRAY_BUFFER, _vbPointsCol);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _pointsCol->size(), _pointsCol->data(), GL_STATIC_DRAW);

	// enable color attribute shader input
	glVertexAttribPointer(_colAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(_colAttribute);
	
	// bind buffer and allocate point size data 
	glBindBuffer(GL_ARRAY_BUFFER, _vbPointsSize);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _pointsSize.size(), _pointsSize.data(), GL_STATIC_DRAW);

	// enable size attribute shader input
	glVertexAttribPointer(_widthAttribute, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glEnableVertexAttribArray(_widthAttribute);

	// /////////////// //
	// lines rendering //
	// /////////////// //
	// create buffers
	glGenVertexArrays(1, &_vaLines);
	glGenBuffers(1, &_vbLinesPos);
	glGenBuffers(1, &_vbLinesCol);
	glGenBuffers(1, &_vbLinesWidth);

	glBindVertexArray(_vaLines);

	// bind buffer and allocate line data (point position)
	glBindBuffer(GL_ARRAY_BUFFER, _vbLinesPos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _knnLinesPos.size(), _knnLinesPos.data(), GL_STATIC_DRAW);

	// enable point attribute shader input
	glVertexAttribPointer(_posAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(_posAttribute);

	// bind buffer and allocate line color data 
	glBindBuffer(GL_ARRAY_BUFFER, _vbLinesCol);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _knnLinesCol.size(), _knnLinesCol.data(), GL_STATIC_DRAW);

	// enable line color attribute shader input
	glVertexAttribPointer(_colAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(_colAttribute);

	// bind buffer and allocate line width data 
	glBindBuffer(GL_ARRAY_BUFFER, _vbLinesWidth);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _knnLinesWidth.size(), _knnLinesWidth.data(), GL_STATIC_DRAW);

	// enable line color attribute shader input
	glVertexAttribPointer(_widthAttribute, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glEnableVertexAttribArray(_widthAttribute);

	// ///////////////////// //
	// single line rendering //
	// ///////////////////// //
	// create buffers
	glGenVertexArrays(1, &_vaLineSingle);
	glGenBuffers(1, &_vbLineSinglePos);
	glGenBuffers(1, &_vbLineSingleCol);
	glGenBuffers(1, &_vbLineSingleWidth);

	glBindVertexArray(_vaLineSingle);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLineSinglePos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _lineSinglePos.size(), _lineSinglePos.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(_posAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(_posAttribute);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLineSingleCol);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _lineSingleCol.size(), _lineSingleCol.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(_colAttribute, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(_colAttribute);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLineSingleWidth);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _lineSingleWidth.size(), _lineSingleWidth.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(_widthAttribute, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glEnableVertexAttribArray(_widthAttribute);

	// reset all buffer bindings
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Renderer::render()
{
	double currentFrame(0);

	int64_t nodeA(0), nodeB(0);
	int64_t knnNew(10), knnMin(1), knnMax(100);
	float noiseLevel(0.4f), knnOpacity(0.5f);
	bool drawKnn(false);
	bool renewData(false), renewKnn(false), renewShortestPath(false), renewNodes(false), renewRandomWalks(false);
	utils::TestData dataSelection(utils::TestData::SwissRole);

	auto newData = [this, &dataSelection, &noiseLevel]() -> void {
		switch (dataSelection) {
		case utils::TestData::SwissRole:
			utils::createSwissRole(_pointsPos, *_pointsCol, dataSize, noiseLevel);
			break;
		case utils::TestData::SCurve: 
			utils::createSCurve(_pointsPos, *_pointsCol, dataSize, noiseLevel);
			break;
		case utils::TestData::Gaussians3D:
			utils::create3dGaussians(_pointsPos, *_pointsCol, dataSize, noiseLevel, 1234, vis::gaussian3DCenters);
			break;
		}

		// normalize to [-1, 1]
		utils::normalizeScale(_pointsPos.begin(), _pointsPos.end());

		_pointsSize.resize(dataSize);
		std::fill(_pointsSize.begin(), _pointsSize.end(), pointSize);
	};

	auto newKnn = [this](int64_t newK) -> void {
		_knnGraph = vis::computeKnn(_data, newK);

		// extract knn graph lines and set colors
		vis::appendKnnLines(_data, _knnGraph.knnIndices, _knnGraph.getK(), _knnLinesPos, _knnLinesCol);
		rebindPointData();

		auto numLines = _knnLinesPos.size() / 3;
		_knnLinesWidth.resize(numLines);
		std::fill(_knnLinesWidth.begin(), _knnLinesWidth.end(), knnLinesWidth);
	};

	auto newShortestPath = [this](int64_t& nodeA, int64_t& nodeB, bool renewNodes) -> void {
		vis::shortestPathNodes(_data, _knnGraph.getGraphView(), geodesicLineColRGB, _geodesicLinesPos, _geodesicLinesCol, nodeA, nodeB, renewNodes);
		_geodesicLinesWidth.resize(_geodesicLinesPos.size() / 3);
		std::fill(_geodesicLinesWidth.begin(), _geodesicLinesWidth.end(), geodesicLinesWidth);
	};

	auto newRandomWalks = [this]() -> void {
		vis::doRandomWalk(_knnGraph, randomWalkNum, randomWalkLength, gaussianNormDist, walkStepWeight, _similarities, _randomWalks, randomWalkMaxVal);

		if (selectedDataID >= 0)
		{
			vis::assignRandomWalkColors(this, tnycolormap, randomWalkMaxVal, selectedDataID);
			rebindPointData();
			vis::assignKnnLineWidth(knnLineWeighting, knnLinesWidth, getSimilarities(), getKnnDists(), getKnnIdx(), knnLinesWidth, selectedDataID, getKnNumber(), getKnnLinesWidth());
		}
	};

	// Prepare knn and random walks
	Log::info("Prepare knn");
	newKnn(knnNew);

	/// /////////// ///
	/// RENDER LOOP ///
	/// /////////// ///
	Log::info("Begin render loop");
	while (!glfwWindowShouldClose(_window))
	{
		// per-frame time logic
		currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		if(idleRender)
			glfwWaitEventsTimeout(waitTimeout);
		else
			glfwPollEvents();

		// movement
		moveCameraPos(_window);

		// clear buffer
		glClearColor(backgroundRGB[0], backgroundRGB[1], backgroundRGB[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/// ///// ///
		/// IMGUI ///
		/// ///// ///

		// Reset settings
		renewKnn = false;
		renewData = false;
		renewShortestPath = false;
		renewNodes = false;
		renewRandomWalks = false;
		isMenuHovered = false;

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		//ImGui::ShowDemoWindow();
		ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_MenuBar);

		if (ImGui::IsWindowHovered()) {
			isMenuHovered = true;
		}

		if (ImGui::TreeNode("Render settings"))
		{
			ImGui::SliderFloat("Camera speed", &cameraSpeed, 0.f, 0.1f);
			ImGui::Checkbox("Idle", &idleRender);

			ImGui::TreePop();
		} // ImGui::TreeNode: Render settings

		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::TreeNode("Data settings"))
		{
			if (ImGui::Combo("Data", (int*)&dataSelection, "Swiss Roll\0S Curve\0Gaussians 3D\0"))
			{
				noiseLevel = vis::testDataNoise(dataSelection);

				renewData = true;
				renewKnn = true;
				renewShortestPath = true;
				renewRandomWalks = drawRandomWalks;
			}

			if (ImGui::SliderInt("Num. data points", &dataSize, 100, 5000))
			{
				dataViewSize = dataSize;

				if (nodeA >= dataSize || nodeB >= dataSize)
				{
					nodeA = 0;
					nodeB = 0;
					Log::warn("Node point larger than new data size of {}, reset node points to 0", dataSize);
				}

				renewData = true;
				renewKnn = true;
				renewShortestPath = true;
				renewRandomWalks = drawRandomWalks;
			}

			if (ImGui::SliderFloat("Noise", &noiseLevel, 0.f, 1.f))
			{
				renewData = true;
				renewKnn = true;
				renewShortestPath = true;
				renewRandomWalks = drawRandomWalks;
			}

			if (ImGui::SliderScalar("Num. kNN", ImGuiDataType_U64, &knnNew, &knnMin, &knnMax))
			{
				renewKnn = true;
				renewShortestPath = true;
				renewRandomWalks = drawRandomWalks;
			}

			ImGui::TreePop();
		} // ImGui::TreeNode: Data settings

		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::TreeNode("Visual settings"))
		{
			if (ImGui::SliderFloat("Point size", &pointSize, 0.f, 20.f))
			{
				std::fill(_pointsSize.begin(), _pointsSize.end(), pointSize);
				sphere_radius = .0005f * pointSize;
				sphere_radius_squared = sphere_radius * sphere_radius;
				rebindPointData();
			}

			ImGui::Checkbox("Draw kNN connections", &drawKnn);
			ImGui::SliderFloat("kNN opacity", &knnOpacity, 0.f, 1.f);
			if (ImGui::SliderFloat("kNN line width", &knnLinesWidth, 0.f, 10.f))
				vis::assignKnnLineWidth(knnLineWeighting, knnLinesWidth, getSimilarities(), getKnnDists(), getKnnIdx(), knnLinesWidth, selectedDataID, getKnNumber(), getKnnLinesWidth());

			if (ImGui::SliderFloat("Geodesic line width", &geodesicLinesWidth, 0.f, 20.f))
				std::fill(_geodesicLinesWidth.begin(), _geodesicLinesWidth.end(), geodesicLinesWidth);

			if (ImGui::ColorEdit3("Geodesic color", geodesicLineColRGB.data()))
			{
				SPH_PARALLEL
				for (size_t i = 0; i < _geodesicLinesCol.size(); i += 3) {
					_geodesicLinesCol[i] = geodesicLineColRGB[0];
					_geodesicLinesCol[i + 1] = geodesicLineColRGB[1];
					_geodesicLinesCol[i + 2] = geodesicLineColRGB[2];
				}
			}

			ImGui::SliderFloat("Euclid line width", &singleLineWidth, 0.f, 20.f);
			ImGui::ColorEdit3("Euclid color", euclideanLineColRGB.data());

			ImGui::ColorEdit3("Background color", backgroundRGB.data());
			ImGui::SliderFloat("Point opacity", &pointOpacity, 0.f, 1.0f);

			ImGui::TreePop();
		} // ImGui::TreeNode: Visual settings

		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::TreeNode("Compute settings"))
		{
			if (ImGui::Button("Compute path between random nodes"))
			{
				renewShortestPath = true;
				renewNodes = true;
			}
			ImGui::SameLine();
			ImGui::Text("Nodes: %i and %i", static_cast<int>(nodeA), static_cast<int>(nodeB));

			if (ImGui::Checkbox("Compute random walk on click", &drawRandomWalks))
			{
				renewRandomWalks = true;

				if (!drawRandomWalks)
				{
					useDataPointColor();
					rebindPointData();
					knnLineWeighting = 0;
					std::fill(_knnLinesWidth.begin(), _knnLinesWidth.end(), knnLinesWidth);
				}
				else
					knnLineWeighting = 1;

			}

			renewRandomWalks |= ImGui::SliderInt("Number of walks", &randomWalkNum, 0, 250);
			renewRandomWalks |= ImGui::SliderInt("Length of walk", &randomWalkLength, 0, 250);

			{
				ImGui::PushID("StepCountWeight");
				ImGui::Text("Step count weight:"); ImGui::SameLine();
				renewRandomWalks |= ImGui::RadioButton("Equal",		&walkStepWeight, 0); ImGui::SameLine();
				renewRandomWalks |= ImGui::RadioButton("Linear",	&walkStepWeight, 1); ImGui::SameLine();
				renewRandomWalks |= ImGui::RadioButton("Gaussian",	&walkStepWeight, 2);
				ImGui::PopID();
			}

			{
				ImGui::PushID("DistanceNorm");
				ImGui::Text("Distance norm:"); ImGui::SameLine();
				renewRandomWalks |= ImGui::RadioButton("Linear",		&gaussianNormDist, 0); ImGui::SameLine();
				renewRandomWalks |= ImGui::RadioButton("Gaussian",		&gaussianNormDist, 1);
				ImGui::PopID();
			}

			ImGui::Text("Random walk values: 0 "); ImGui::SameLine();
			if (ImPlot::ColormapButton("similarity", ImVec2(225, 0), impltcolormap))
			{
				currentColormap = (currentColormap + 1) % vis::colormaps.size();
				std::tie(impltcolormap, tnycolormap) = vis::getColorMapPair(currentColormap);

				if (selectedDataID >= 0)
				{
					useRandomWalkPointColor();
					vis::assignRandomWalkColors(this, tnycolormap, randomWalkMaxVal, selectedDataID);
					rebindPointData();
				}
			}
			ImGui::SetItemTooltip("Color mapping not linear but with sqrt()");
			ImGui::SameLine(); ImGui::Text(" %f", randomWalkMaxVal);

			bool radioToggled = false;
			ImGui::Text("kNN line weight:"); ImGui::SameLine();
			radioToggled |= ImGui::RadioButton("Distance", &knnLineWeighting, 1); ImGui::SameLine();
			ImGui::SetItemTooltip("Are multiplied by 2 times the line weight");
			radioToggled |= ImGui::RadioButton("Similarity", &knnLineWeighting, 2); ImGui::SameLine();
			ImGui::SetItemTooltip("Normalized distances using gaussian weighting, \nas done for t-SNE");
			radioToggled |= ImGui::RadioButton("All", &knnLineWeighting, 0);

			if (radioToggled)
			{
				if (selectedDataID >= 0)
					vis::assignKnnLineWidth(knnLineWeighting, knnLinesWidth, getSimilarities(), getKnnDists(), getKnnIdx(), knnLinesWidth, selectedDataID, getKnNumber(), getKnnLinesWidth());
				else
					std::fill(_knnLinesWidth.begin(), _knnLinesWidth.end(), knnLinesWidth);
			}

			ImGui::TreePop();
		} // ImGui::TreeNode: Compute settings

		ImGui::End();

		/// /////// ///
		/// COMPUTE ///
		/// /////// ///

		if (renewData)
			newData();

		if (renewKnn)
			newKnn(knnNew);

		if (renewShortestPath)
			newShortestPath(nodeA, nodeB, renewNodes);

		if (drawRandomWalks && renewRandomWalks)
			newRandomWalks();

		/// /////// ///
		/// DRAWING ///
		/// /////// ///

		projectionMatrix = glm::perspective(glm::radians(fov), static_cast<float>(_windowWidth) / static_cast<float>(_windowWidth), nearClip, farClip);
		viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

		// draw the data points
		drawPoints(pointOpacity);

		// draw knn connections
		if (drawKnn)
			drawLines(_knnLinesPos, _knnLinesCol, _knnLinesWidth, knnOpacity);

		// draw geodesic lines
		if (_geodesicLinesPos.size() >= 3)
		{
			// draw direct connection
			drawLine(nodeA, nodeB, { euclideanLineColRGB[0], euclideanLineColRGB[1], euclideanLineColRGB[2] }, { euclideanLineColRGB[0], euclideanLineColRGB[1], euclideanLineColRGB[2] }, singleLineWidth);

			// draw geodesic path
			if(nodeA != nodeB)
				drawLines(_geodesicLinesPos, _geodesicLinesCol, _geodesicLinesWidth);
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(_window);
	}
}

void Renderer::rebindPointData() const
{
	glBindBuffer(GL_ARRAY_BUFFER, _vbPointsPos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _pointsPos.size(), _pointsPos.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, _vbPointsCol);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _pointsCol->size(), _pointsCol->data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, _vbPointsSize);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _pointsSize.size(), _pointsSize.data(), GL_STATIC_DRAW);
}

void Renderer::drawPoints(float opacity, bool rebind) const
{
	_pointShader->use();
	_pointShader->setMat4("projection", projectionMatrix);
	_pointShader->setMat4("view", viewMatrix);
	_pointShader->setFloat("opacity", opacity);

	glBindVertexArray(_vaPoints);

	if (rebind)
		rebindPointData();

	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(_pointsPos.size() / 3));

	if (rebind)
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}

void Renderer::drawLine(int64_t nodeA, int64_t nodeB, Color colA, Color colB, float lineWidth, float opacity)
{
	// Point positions
	_lineSinglePos[0] = _pointsPos[nodeA * 3 + 0];
	_lineSinglePos[1] = _pointsPos[nodeA * 3 + 1];
	_lineSinglePos[2] = _pointsPos[nodeA * 3 + 2];

	_lineSinglePos[3] = _pointsPos[nodeB * 3 + 0];
	_lineSinglePos[4] = _pointsPos[nodeB * 3 + 1];
	_lineSinglePos[5] = _pointsPos[nodeB * 3 + 2];

	// Point colors
	_lineSingleCol[0] = colA.r;
	_lineSingleCol[1] = colA.g;
	_lineSingleCol[2] = colA.b;

	_lineSingleCol[3] = colB.r;
	_lineSingleCol[4] = colB.g;
	_lineSingleCol[5] = colB.b;

	// widths
	_lineSingleWidth[0] = lineWidth;
	_lineSingleWidth[1] = lineWidth;

	_lineShader->use();
	_lineShader->setMat4("projection", projectionMatrix);
	_lineShader->setMat4("view", viewMatrix);
	_lineShader->setFloat("opacity", opacity);

	glBindVertexArray(_vaLineSingle);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLineSinglePos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, _lineSinglePos.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLineSingleCol);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, _lineSingleCol.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLineSingleWidth);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2, _lineSingleWidth.data(), GL_STATIC_DRAW);

	glDrawArrays(GL_LINES, 0, 2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Renderer::drawLines(std::vector<float>& linePos, std::vector<float>& lineCol, std::vector<float>& lineWidths, float opacity) const
{
	_lineShader->use();
	_lineShader->setMat4("projection", projectionMatrix);
	_lineShader->setMat4("view", viewMatrix);
	_lineShader->setFloat("opacity", opacity);

	glBindVertexArray(_vaLines);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLinesPos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * linePos.size(), linePos.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLinesCol);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * lineCol.size(), lineCol.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, _vbLinesWidth);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * lineWidths.size(), lineWidths.data(), GL_STATIC_DRAW);

	glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(linePos.size() / 3));

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}

/// ///////// ///
/// Callbacks ///
/// ///////// ///

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);

	auto* renderHandle = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
	renderHandle->setWindowWidth(width);
	renderHandle->setWindowHeight(height);
}

static inline glm::vec3 get_world_coords(float mouse_x, float mouse_y, int windowWidth, int windowHeight)
{
	// viewport space to normalized device space
	float normalizedX = (2.0f * mouse_x) / windowWidth - 1.0f;
	float normalizedY = 1.0f - (2.0f * mouse_y) / windowHeight;

	// to homogeneous clip space
	glm::vec4 clipSpace = glm::vec4(normalizedX, normalizedY, -1.0f, 1.0f);

	// to eye space
	glm::vec4 eyeSpace  = glm::inverse(projectionMatrix) * clipSpace;
	eyeSpace = glm::vec4(eyeSpace.x, eyeSpace.y, -1.0f, 0.0f);		// -1.0f: forwards, 0.0f: not a point

	// to world space
	glm::vec4 worldSpace = glm::inverse(viewMatrix) * eyeSpace ;
	worldSpace = glm::normalize(worldSpace);

	return glm::vec3(worldSpace);
}

static inline bool ray_sphere_intersect(glm::vec3 ray_origin_wor, glm::vec3 ray_direction_wor, const std::vector<float>& _pointsPos, int pointID, float* intersection_distance) 
{
	glm::vec3 sphere_centre_word = { _pointsPos[pointID * 3], _pointsPos[pointID * 3 + 1], _pointsPos[pointID * 3 + 2] };
	glm::vec3 dist_to_sphere = ray_origin_wor - sphere_centre_word;

	// solve equations: points on sphere and points on rays intersections
	float b = glm::dot(ray_direction_wor, dist_to_sphere);
	float c = glm::dot(dist_to_sphere, dist_to_sphere) - sphere_radius_squared;
	float b_squared_minus_c = b * b - c;

	// ray completely sphere
	if (b_squared_minus_c < 0.0f) {
		return false; 
	}

	// ray hits twice
	if (b_squared_minus_c > 0.0f) {
		// 2 intersection distances along ray
		float t_a = -b + std::sqrt(b_squared_minus_c);
		float t_b = -b - std::sqrt(b_squared_minus_c);
		*intersection_distance = t_b;

		// if behind viewer, throw one or both away
		if (t_a < 0.0) {
			if (t_b < 0.0) { return false; }
		}
		else if (t_b < 0.0) {
			*intersection_distance = t_a;
		}

		return true;
	}

	// ray hits once
	if (0.0f == b_squared_minus_c) {
		// if behind viewer, throw away
		float t = -b + std::sqrt(b_squared_minus_c);
		if (t < 0.0f) { return false; }
		*intersection_distance = t;
		return true;
	}

	return false;
}

void mouse_click_callback(GLFWwindow* window, int button, int action, [[maybe_unused]] int mods)
{
	// mouse picking with ray casting
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {

		if (isMenuHovered)
			return;

		auto* renderHandle = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));

		if (!renderHandle) {
			Log::warn("No render handle");
			return;
		}

		int windowWidth = renderHandle->getWindowWidth();
		int windowHeight = renderHandle->getWindowHeight();

		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		glm::vec3 rayWorld = get_world_coords(static_cast<float>(xpos), static_cast<float>(ypos), windowWidth, windowHeight);

		// for all data point, check if the mouse ray intersects with them
		int closestSphereID = -1;
		float closest_intersection = .0f;

		for (int i = 0; i < dataSize; i++) {
			float t_dist = 0.0f;
			if (ray_sphere_intersect(cameraPos, rayWorld, renderHandle->getPointPositions(), i, &t_dist)) {
				// if more than one sphere is in path of ray, only use the closest one
				if (-1 == closestSphereID || t_dist < closest_intersection) {
					closestSphereID = i;
					closest_intersection = t_dist;
				}
			}
		}

		if (selectedDataID >= 0)
		{
			renderHandle->useDataPointColor();
			renderHandle->setColor(selectedDataID, prevSelColor.r, prevSelColor.g, prevSelColor.b);
			
			vis::assignKnnLineWidth(0, knnLinesWidth, renderHandle->getSimilarities(), renderHandle->getKnnDists(), renderHandle->getKnnIdx(), knnLinesWidth, closestSphereID, renderHandle->getKnNumber(), renderHandle->getKnnLinesWidth());
		}

		if (closestSphereID >= 0)
		{
			Log::info("Clicked Point: {}", closestSphereID);

			renderHandle->useDataPointColor();
			auto& cols = renderHandle->getPointColors();
			prevSelColor = { cols[closestSphereID * 3], cols[closestSphereID * 3 + 1] , cols[closestSphereID * 3 + 2] };

			if (drawRandomWalks)
			{
				renderHandle->useRandomWalkPointColor();
				vis::assignRandomWalkColors(renderHandle, tnycolormap, randomWalkMaxVal, closestSphereID);
				vis::assignKnnLineWidth(knnLineWeighting, knnLinesWidth, renderHandle->getSimilarities(), renderHandle->getKnnDists(), renderHandle->getKnnIdx(), knnLinesWidth, closestSphereID, renderHandle->getKnNumber(), renderHandle->getKnnLinesWidth());
			}

			renderHandle->setColor(closestSphereID, 1, 1, 1);

		}

		if(selectedDataID >= 0 || closestSphereID >= 0)
			renderHandle->rebindPointData();

		selectedDataID = closestSphereID;
	}

}

///
/// Code below adapted from: https://github.com/JoeyDeVries/LearnOpenGL
/// CC BY-NC 4.0, Joey de Vries, https://learnopengl.com
/// 

// handle every key just once
void key_callbacks(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_ESCAPE)
			glfwSetWindowShouldClose(window, true);

		if (key == GLFW_KEY_T)
			mouseActive = !mouseActive;

	}
}

// query GLFW whether relevant keys are pressed/released this (every) frame and react accordingly
void moveCameraPos(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraLeft)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraLeft)) * cameraSpeed;
}

// glfw: whenever the mouse moves, this callback is called
void mouse_move_callback([[maybe_unused]] GLFWwindow* window, double xposIn, double yposIn)
{
	if (!mouseActive)
	{
		firstMouse = true;
		return;
	}

	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.2f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// make sure that when pitch is out of bounds, screen doesn't get flipped
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front{};
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}

void glfw_error_callback(int error, const char* description)
{
	Log::error("GLFW Error {0}: {1}", error, description);
}
