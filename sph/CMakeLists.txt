cmake_minimum_required(VERSION 3.30...3.31)

# -----------------------------------------------------------------------------
#  Main Library
# -----------------------------------------------------------------------------

message(STATUS "Configuring ${SPH_LIB} target")

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(SPH_COMPUTE_HEADERS
    NearestNeighbors.hpp
    ImageHierarchy.hpp
    LevelSimilarities.hpp
)

set(SPH_COMPUTE_SOURCES
    NearestNeighbors.cpp
    ImageHierarchy.cpp
    LevelSimilarities.cpp
)

set(SPH_COMPUTE_TSNE_HEADERS
    EmbedTsne.hpp
)

set(SPH_COMPUTE_TSNE_SOURCES
    EmbedTsne.cpp
)

set(SPH_COMPUTE_UMAP_HEADERS
    EmbedUmap.hpp
)

set(SPH_COMPUTE_UMAP_SOURCES
    EmbedUmap.cpp
)

set(SPH_COMPUTE_WRAPPER_HEADERS
    ComputeHierarchy.hpp
    ComputeEmbedding.hpp
)

set(SPH_COMPUTE_WRAPPER_SOURCES
    ComputeHierarchy.cpp
    ComputeEmbedding.cpp
)

list(APPEND SPH_COMPUTE_HEADERS ${SPH_COMPUTE_TSNE_HEADERS} ${SPH_COMPUTE_UMAP_HEADERS} ${SPH_COMPUTE_WRAPPER_HEADERS})
list(APPEND SPH_COMPUTE_SOURCES ${SPH_COMPUTE_TSNE_SOURCES} ${SPH_COMPUTE_UMAP_SOURCES} ${SPH_COMPUTE_WRAPPER_SOURCES})

set(UTILS_HEADERS
    utils/ShortestPath.hpp
    utils/Distances.hpp
    utils/DistanceCache.hpp
    utils/Logger.hpp
    utils/Timer.hpp
    utils/FileIO.hpp
    utils/EvalIO.hpp
    utils/Algorithms.hpp
    utils/Similarities.hpp
    utils/CommonDefinitions.hpp
    utils/ProgressBar.hpp
    utils/MaxSizeDeque.hpp
    utils/Math.hpp
    utils/SparseMatrixAlgorithms.hpp
    utils/GraphNormalization.hpp
    utils/Statistics.hpp
    utils/AStar.hpp
    utils/AStarBoost.hpp
    utils/Settings.hpp
    utils/Scaler.hpp
    utils/Knn.hpp
    utils/PCA.hpp
)

set(UTILS_SOURCES
    utils/ShortestPath.cpp
    utils/Logger.cpp
    utils/FileIO.cpp
    utils/EvalIO.cpp
    utils/Similarities.cpp
    utils/Math.cpp
    utils/SparseMatrixAlgorithms.cpp
    utils/GraphNormalization.cpp
    utils/Statistics.cpp
    utils/AStar.cpp
    utils/AStarBoost.cpp
    utils/Settings.cpp
    utils/Scaler.cpp
    utils/Knn.cpp
)

set(UTILS_DATASTRUCTURES_HEADERS
    utils/Hierarchy.hpp
    utils/Graph.hpp
    utils/GraphBoost.hpp
    utils/GraphUtils.hpp
    utils/GraphAdapterBoost.hpp
    utils/Data.hpp
    utils/Cacheable.hpp
    utils/Embedding.hpp
    utils/Histogram.hpp
    utils/MatrixCSR.hpp
)

set(UTILS_DATASTRUCTURES_SOURCES
    utils/Hierarchy.cpp
    utils/Graph.cpp
    utils/GraphUtils.cpp
    utils/Data.cpp
    utils/Cacheable.cpp
    utils/Embedding.cpp
    utils/MatrixCSR.cpp
)

set(UTILS_MISC_HEADERS
    utils/ImageHelper.hpp
    utils/PrintHelper.hpp
    utils/TestData.hpp
    utils/HDILibHelper.hpp
)

set(UTILS_MISC_SOURCES
    utils/ImageHelper.cpp
    utils/PrintHelper.cpp
    utils/TestData.cpp
)

set(UTILS_SPACE_HEADERS
    utils/NeighborOverlapSpace.hpp
    utils/NeighborWalksSingleOverlapSpace.hpp
    utils/NeighborWalksBhattacharyyaSpace.hpp
    utils/GeodesicPathSpace.hpp
    utils/EuclidDistSpace.hpp
)

set(SPH_SOURCES ${SPH_COMPUTE_SOURCES} ${UTILS_SOURCES} ${UTILS_DATASTRUCTURES_SOURCES} ${UTILS_MISC_SOURCES})
set(SPH_HEADERS ${SPH_COMPUTE_HEADERS} ${UTILS_HEADERS} ${UTILS_SPACE_HEADERS} ${UTILS_DATASTRUCTURES_HEADERS} ${UTILS_MISC_HEADERS})

source_group(Compute FILES ${SPH_COMPUTE_HEADERS} ${SPH_COMPUTE_SOURCES})
source_group(Utils FILES ${UTILS_HEADERS} ${UTILS_SOURCES})
source_group(Utils/Spaces FILES ${UTILS_SPACE_HEADERS})
source_group("Utils/Data structures" FILES ${UTILS_DATASTRUCTURES_HEADERS} ${UTILS_DATASTRUCTURES_SOURCES})
source_group(Utils/Misc FILES ${UTILS_MISC_HEADERS} ${UTILS_MISC_SOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_library(${SPH_LIB} ${SPH_HEADERS} ${SPH_SOURCES} ${SPH_GPU_SOURCES} ${SPH_GPU_HEADERS})

# Export SPH_ variable to parent scope.
set(SPH_HEADERS ${SPH_HEADERS} ${SPH_GPU_HEADERS} PARENT_SCOPE)

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------

# Include library headers
target_include_directories(${SPH_LIB} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# Include third party headers
target_include_directories(${SPH_LIB} PUBLIC  "${HDILib_SOURCE_DIR}")
target_include_directories(${SPH_LIB} PRIVATE ${TINYCOLORMAP_INCLUDE_DIRS})

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------

target_link_libraries(${SPH_LIB} PRIVATE ${faiss_target})
target_link_libraries(${SPH_LIB} PRIVATE range-v3::range-v3) # range-v3::meta range-v3::concepts
target_link_libraries(${SPH_LIB} PRIVATE hnswlib::hnswlib)
target_link_libraries(${SPH_LIB} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${SPH_LIB} PRIVATE libscran::umappp)
target_link_libraries(${SPH_LIB} PRIVATE TIFF::TIFF)
target_link_libraries(${SPH_LIB} PRIVATE PNG::PNG)
target_link_libraries(${SPH_LIB} PRIVATE JPEG::JPEG)
target_link_libraries(${SPH_LIB} PRIVATE libscran::umappp)

target_link_libraries(${SPH_LIB} PUBLIC  Eigen3::Eigen)
target_link_libraries(${SPH_LIB} PUBLIC  Boost::graph)
target_link_libraries(${SPH_LIB} PUBLIC  hdidimensionalityreduction hdidata hdiutils)
target_link_libraries(${SPH_LIB} PUBLIC  spdlog::spdlog_header_only)
target_link_libraries(${SPH_LIB} PUBLIC  fmt::fmt-header-only)
target_link_libraries(${SPH_LIB} PUBLIC  nlohmann_json::nlohmann_json)
target_link_libraries(${SPH_LIB} PUBLIC  unordered_dense::unordered_dense)

if(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(${SPH_LIB} PRIVATE TBB::tbb)
endif()

if(${SPH_BUILD_RUNTIME_GPU})
    target_link_libraries(${SPH_LIB} PRIVATE glfw)
    target_link_libraries(${SPH_LIB} PRIVATE ${OPENGL_LIBRARIES})
    message(STATUS "Defining symbol __RUNTIME_GPU__ in target ${SPH_LIB}")
    target_compile_definitions (${SPH_LIB} PRIVATE __RUNTIME_GPU__)
endif()

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------

# Request C++20
target_compile_features(${SPH_LIB} PRIVATE cxx_std_20)

# Instruction sets and optimization
sph_check_and_set_AVX(${SPH_LIB} ${SPH_USE_AVX})
sph_set_optimization_level(${SPH_LIB} ${SPH_OPTIMIZATION_LEVEL})

set_target_properties(${SPH_LIB} PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(${SPH_LIB} PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Misc
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${SPH_LIB} PRIVATE /W3)
    target_compile_options(${SPH_LIB} PRIVATE /bigobj)                          # for Eigen PCA
    target_compile_options(${SPH_LIB} PRIVATE $<$<CONFIG:Debug>:/RTC1>)         # Run-time error checks
    target_compile_definitions(${SPH_LIB} PRIVATE _CRT_SECURE_NO_WARNINGS)      # libpng and libjpeg make use use std::fopen

    target_compile_options(${SPH_LIB} PRIVATE ${MSVC_LINKAGE_TYPE})

else()
    target_compile_options(${SPH_LIB} PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Precompiled headers
if(${SPH_PRECOM_HEADER})
    get_target_property(nlohmann_json_include_dirs nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    list(GET nlohmann_json_include_dirs 0 nlohmann_json_include_dir)

    set(DEP_PRECOM_HEADERS
        ${nlohmann_json_include_dir}/nlohmann/json.hpp
    )

    set(SPH_PRECOM_HEADERS
        utils/Algorithms.hpp
        utils/Cacheable.hpp
        utils/CommonDefinitions.hpp
        utils/Data.hpp
        utils/DistanceCache.hpp
        utils/Distances.hpp
        utils/FileIO.hpp
        utils/Graph.hpp
        utils/Logger.hpp
        utils/ProgressBar.hpp
        utils/Settings.hpp
        utils/Timer.hpp
    )

    target_precompile_headers(${SPH_LIB}
        PRIVATE
        ${DEP_PRECOM_HEADERS}
        ${SPH_PRECOM_HEADERS}
    )
endif()

if (APPLE)
    target_compile_definitions(${SPH_LIB} PRIVATE -DGL_SILENCE_DEPRECATION)
endif()
