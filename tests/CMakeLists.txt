cmake_minimum_required(VERSION 3.30...3.31)

# -----------------------------------------------------------------------------
#  Unit tests
# -----------------------------------------------------------------------------

set(SPH_TESTS "SPHTests")
message(STATUS "Configuring ${SPH_TESTS} target")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(Catch2 CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(TEST_FILES
    Tests.cpp
    AStarTest.hpp
    AStarTest.cpp
    HierarchyTest.hpp
    HierarchyTest.cpp
    UtilsTests.hpp
    UtilsTests.cpp
)

set(BENCH_FILES
    UtilsBenchmark.hpp
    UtilsBenchmark.cpp
)

set(HELPER_FILES
    HelperCatch2.hpp
    HelperSPH.hpp
)

set(SOURCES ${TEST_FILES} ${BENCH_FILES} ${HELPER_FILES})

source_group(Tests FILES ${TEST_FILES})
source_group(Bench FILES ${BENCH_FILES})
source_group(Helper FILES ${HELPER_FILES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_executable(${SPH_TESTS} ${SOURCES})

add_test(
    NAME ${SPH_TESTS}
    COMMAND $<TARGET_FILE:${SPH_TESTS}>
)

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------

target_link_libraries(${SPH_TESTS} PRIVATE ${SPH_LIB})
target_link_libraries(${SPH_TESTS} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${SPH_TESTS} PRIVATE Catch2::Catch2WithMain)

if(${SPH_BUILD_RUNTIME_GPU})
    target_compile_definitions(${SPH_TESTS} PRIVATE __RUNTIME_GPU__)
endif()

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------

# Request C++20
target_compile_features(${SPH_TESTS} PRIVATE cxx_std_20)

# Instruction sets
sph_check_and_set_AVX(${SPH_TESTS} ${SPH_USE_AVX})
sph_set_optimization_level(${SPH_TESTS} ${SPH_OPTIMIZATION_LEVEL})

set_target_properties(${SPH_TESTS} PROPERTIES DEBUG_POSTFIX "d")

# Misc
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${SPH_TESTS} PRIVATE /W3)
    target_compile_options(${SPH_TESTS} PRIVATE $<$<CONFIG:Debug>:/RTC1>)        # Run-time error checks
    target_compile_options(${SPH_TESTS} PRIVATE ${MSVC_LINKAGE_TYPE})
else()
    target_compile_options(${SPH_TESTS} PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Precompiled headers
if(${SPH_PRECOM_HEADER})
    get_target_property(Catch2_include_dirs Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
    list(GET Catch2_include_dirs 0 Catch2_include_dir)

    set(DEP_PRECOM_HEADERS_TEST
        ${Catch2_include_dir}/catch2/catch_session.hpp
        ${Catch2_include_dir}/catch2/catch_test_macros.hpp
        ${Catch2_include_dir}/catch2/matchers/catch_matchers.hpp
        ${Catch2_include_dir}/catch2/matchers/catch_matchers_floating_point.hpp
        ${Catch2_include_dir}/catch2/benchmark/catch_benchmark.hpp
    )

    target_precompile_headers(${SPH_TESTS}
        PRIVATE
        ${DEP_PRECOM_HEADERS_TEST}
    )
endif()

if (APPLE)
    target_compile_definitions(${SPH_TESTS} PRIVATE -DGL_SILENCE_DEPRECATION)
endif()
