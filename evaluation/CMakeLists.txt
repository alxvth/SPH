cmake_minimum_required(VERSION 3.30...3.31)

# -----------------------------------------------------------------------------
#  Evaluation
# -----------------------------------------------------------------------------

set(SPH_EVAL "SPHEvaluation")
message(STATUS "Configuring ${SPH_EVAL} target")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(SPH_EVAL_FILES
    Evaluation.cpp
    RunEvaluation.hpp
    RunEvaluation.cpp
)

set(UTIL_FILES
    EvaluationSettings.hpp
    EvaluationSettings.cpp
)

set(EVAL_SOURCES ${SPH_EVAL_FILES} ${UTIL_FILES})

source_group(Evaluation FILES ${SPH_EVAL_FILES})
source_group(Utils FILES ${UTIL_FILES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_executable(${SPH_EVAL} ${EVAL_SOURCES})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------

target_link_libraries(${SPH_EVAL} PRIVATE ${SPH_LIB})
target_link_libraries(${SPH_EVAL} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${SPH_EVAL} PRIVATE range-v3::range-v3) # range-v3::meta range-v3::concepts

if(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(${SPH_EVAL} PRIVATE TBB::tbb)
endif()

if(${SPH_BUILD_RUNTIME_GPU})
    target_link_libraries(${SPH_EVAL} PRIVATE glfw)
    message(STATUS "Defining symbol __RUNTIME_GPU__ in target ${SPH_EVAL}")
    target_compile_definitions (${SPH_EVAL} PRIVATE __RUNTIME_GPU__)
endif()

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------

# Request C++20
target_compile_features(${SPH_EVAL} PRIVATE cxx_std_20)

# Instruction sets
sph_check_and_set_AVX(${SPH_EVAL} ${SPH_USE_AVX})
sph_set_optimization_level(${SPH_EVAL} ${SPH_OPTIMIZATION_LEVEL})

set_target_properties(${SPH_EVAL} PROPERTIES DEBUG_POSTFIX "d")

# Misc
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${SPH_EVAL} PRIVATE /W3)
    target_compile_options(${SPH_EVAL} PRIVATE ${MSVC_LINKAGE_TYPE})
else()
    target_compile_options(${SPH_EVAL} PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

if (APPLE)
    target_compile_definitions(${SPH_EVAL} PRIVATE -DGL_SILENCE_DEPRECATION)
endif()

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
set(SPH_EVAL_INSTALL_COMPONENT ${SPH_EVAL}_install)
set(SPH_EVAL_INSTALL_FOLDER evaluation)

install(TARGETS ${SPH_EVAL}
        RUNTIME DESTINATION ${SPH_EVAL_INSTALL_FOLDER}
        COMPONENT ${SPH_EVAL_INSTALL_COMPONENT}
)

install_dependencies(${SPH_EVAL} ${SPH_EVAL_INSTALL_COMPONENT} ${SPH_EVAL_INSTALL_FOLDER} ${SPH_LIB})

add_custom_target(install_evaluation
    COMMAND ${CMAKE_COMMAND} 
        --install ${CMAKE_CURRENT_BINARY_DIR} 
        --config $<CONFIGURATION> 
        --component ${SPH_EVAL_INSTALL_COMPONENT}
)

add_dependencies(install_evaluation ${SPH_EVAL})
