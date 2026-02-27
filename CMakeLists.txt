cmake_minimum_required(VERSION 3.31)

# -----------------------------------------------------------------------------
# User Options
# -----------------------------------------------------------------------------
option(SPH_USE_AVX "Use AVX if available" ON)
option(SPH_BUILD_VIS "Build a visualization project to showcase and test features of the Spatial Hierarchy Library" OFF)
option(SPH_BUILD_TESTS "Build unit tests for Spatial Hierarchy Library" OFF)
option(SPH_BUILD_EVAL "Build evaluation of the Spatial Hierarchy Library" OFF)
option(SPH_BUILD_RUNTIME_GPU "Build runtime GPU features" ON)
option(SPH_PRECOM_HEADER "Use precompiled headers" ON)
option(SPH_LOCAL_TBB "Use TBB from vcpkg" OFF)
set(SPH_OPTIMIZATION_LEVEL "2" CACHE STRING "Optimization level for all SPH targets in release builds, e.g. 0, 1, 2")

if(SPH_BUILD_VIS)
    list(APPEND VCPKG_MANIFEST_FEATURES "visualization")
    message(STATUS "Adding to config: VIS")
endif()

if(SPH_BUILD_TESTS)
    list(APPEND VCPKG_MANIFEST_FEATURES "unittests")
    message(STATUS "Adding to config: TESTS")
endif()

if(SPH_BUILD_RUNTIME_GPU)
    list(APPEND VCPKG_MANIFEST_FEATURES "runtimegpu")
    message(STATUS "Adding to config: RUNTIMEGPU")
endif()

if(SPH_BUILD_EVAL)
    list(APPEND VCPKG_MANIFEST_FEATURES "evaluation")
    message(STATUS "Adding to config: EVALUATION")
endif()

if(SPH_LOCAL_TBB)
    list(APPEND VCPKG_MANIFEST_FEATURES "local-tbb")
    message(STATUS "Build TBB with vcpkg")
endif()

# vcpkg library settings
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg-overlays")

if(WIN32 AND NOT DEFINED VCPKG_LIBRARY_LINKAGE)
    set(VCPKG_HOST_TRIPLET "x64-windows-static-md" CACHE STRING "")
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "")
    set(VCPKG_LIBRARY_LINKAGE "static" CACHE STRING "")
endif()

# -----------------------------------------------------------------------------
# Project: Spatial Hierarchy Library
# -----------------------------------------------------------------------------

PROJECT("SpatialHierarchyLibrary"
    VERSION 0.0.1
    DESCRIPTION "Hierarchical superpixel for high-dimensional images"
    LANGUAGES CXX)

# -----------------------------------------------------------------------------
# CMake flags, settings and options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(SPH_CMAKE_MODULES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${SPH_CMAKE_MODULES_PATH})

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/EHsc /MP /permissive- /Zc:__cplusplus /Zc:preprocessor)

    set(MSVC_LINKAGE_TYPE $<IF:$<CONFIG:DEBUG>,/MDd,/MD>)
    if(VCPKG_TARGET_TRIPLET STREQUAL "x64-windows-static")
        set(MSVC_LINKAGE_TYPE $<IF:$<CONFIG:DEBUG>,/MTd,/MT>)
    endif()

endif()

if(SPH_BUILD_TESTS)
    include(CTest)
    enable_testing()
endif()

# Test hardware avx capabilities
include(CMakeCheckSetAVX)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|Arm|ARM)")
    set(SPH_USE_AVX OFF CACHE BOOL "Enable some feature for ARM architecture" FORCE)
    message(STATUS "On Apple Silicon AVX instructions are not available, setting SPH_USE_AVX to ${SPH_USE_AVX}.")
endif()

# CMake package manager
include(get_cpm)

# re-applying patches is problematic without CPM_SOURCE_CACHE
# see https://github.com/cpm-cmake/CPM.cmake/issues/577
set(CPM_SOURCE_CACHE ${CMAKE_CURRENT_BINARY_DIR}/.cpm-cache)

# helper for installing dynamic lib dependencies
include(InstallDependencies)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(OpenMP_RUNTIME_MSVC "llvm" FORCE)
endif()
find_package(OpenMP REQUIRED)

find_path(TINYCOLORMAP_INCLUDE_DIRS "tinycolormap.hpp")
find_package(Annoy CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS graph)
find_package(fmt CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)
find_package(libscran_umappp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(TIFF REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(faiss CONFIG REQUIRED)

set(faiss_target "faiss")
if(${SPH_USE_AVX})
  set(faiss_target "faiss_avx2")
endif()

if(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(SPH_LOCAL_TBB)
        find_package(TBB CONFIG REQUIRED)
    else()
        find_package(TBB REQUIRED)
    endif()
    
    find_package(Threads REQUIRED)
endif()

if(${SPH_BUILD_RUNTIME_GPU})
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
endif()

set(HDILib_INSTALL OFF)
set(HDILib_ENABLE_AVX ${SPH_USE_AVX})
set(HDILib_ENABLE_PID ON)
set(HDILib_OPTIMIZATION_LEVEL ${SPH_OPTIMIZATION_LEVEL})

CPMAddPackage(
 NAME              HDILib
 GIT_REPOSITORY    https://github.com/biovault/HDILib.git
 GIT_TAG           v1.2.11 
 EXCLUDE_FROM_ALL  YES
 OPTIONS "HDILib_INSTALL ${HDILib_INSTALL}"
         "HDILib_ENABLE_AVX ${HDILib_ENABLE_AVX}"
         "HDILib_ENABLE_PID ${HDILib_ENABLE_PID}"
         "HDILib_OPTIMIZATION_LEVEL ${HDILib_OPTIMIZATION_LEVEL}"
)

set(HDILib_SOURCE_DIR "${HDILib_SOURCE_DIR}" CACHE STRING "Dependency module source path")
message(STATUS "HDILib_SOURCE_DIR: ${HDILib_SOURCE_DIR}")

set_target_properties(hdidata hdidimensionalityreduction hdiutils
    PROPERTIES 
    FOLDER HDILib
)

if (APPLE)
    target_compile_definitions(hdidimensionalityreduction PRIVATE -DGL_SILENCE_DEPRECATION)
    target_compile_definitions(hdidata PRIVATE -DGL_SILENCE_DEPRECATION)
    target_compile_definitions(hdiutils PRIVATE -DGL_SILENCE_DEPRECATION)
endif()

# -----------------------------------------------------------------------------
# Main target
# -----------------------------------------------------------------------------

# Define main library name at top scope
set(SPH_LIB "SPHLibrary")
add_subdirectory(sph)

# -----------------------------------------------------------------------------
# Other targets
# -----------------------------------------------------------------------------

if(SPH_BUILD_VIS)
    add_subdirectory(vis)
endif()

if(SPH_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(SPH_BUILD_EVAL)
    add_subdirectory(evaluation)
endif()
