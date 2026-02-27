cmake_minimum_required(VERSION 3.30...3.31)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

set(SPH_VIS "SPHVisualization")
message(STATUS "Configuring ${SPH_VIS} target")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(SPH_VIS_HEADERS
    Renderer.hpp
    Shader.hpp
    UtilsCompute.hpp
    UtilsDefaults.hpp
)

set(SPH_VIS_SOURCES
    Vis.cpp
    Renderer.cpp
    Shader.cpp
    UtilsCompute.cpp
)

set(SPH_VIS_RESOURCES
    res/Point.frag
    res/Point.vert
    res/Lines.vert
    res/Lines.geom
)

set(VIS_HEADERS ${SPH_VIS_HEADERS})
set(VIS_SOURCES ${SPH_VIS_SOURCES} ${SPH_VIS_RESOURCES})

source_group(SPHVis FILES ${SPH_VIS_HEADERS} ${SPH_VIS_SOURCES})
source_group(Res FILES ${SPH_VIS_RESOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_executable(${SPH_VIS} ${VIS_SOURCES} ${VIS_HEADERS})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPH_VIS} PRIVATE ${TINYCOLORMAP_INCLUDE_DIRS})

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------

target_link_libraries(${SPH_VIS} PRIVATE ${SPH_LIB})
target_link_libraries(${SPH_VIS} PRIVATE glad::glad)
target_link_libraries(${SPH_VIS} PRIVATE glfw)
target_link_libraries(${SPH_VIS} PRIVATE glm::glm)
target_link_libraries(${SPH_VIS} PRIVATE imgui::imgui)
target_link_libraries(${SPH_VIS} PRIVATE implot::implot)
target_link_libraries(${SPH_VIS} PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------

# Request C++20
target_compile_features(${SPH_VIS} PRIVATE cxx_std_20)

# Instruction sets and optimization
sph_check_and_set_AVX(${SPH_VIS} ${SPH_USE_AVX})
sph_set_optimization_level(${SPH_VIS} ${SPH_OPTIMIZATION_LEVEL})

set_target_properties(${SPH_VIS} PROPERTIES DEBUG_POSTFIX "d")

# Warning levels
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${SPH_VIS} PRIVATE /W3)
    target_compile_options(${SPH_VIS} PRIVATE ${MSVC_LINKAGE_TYPE})
else()
    target_compile_options(${SPH_VIS} PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

if (APPLE)
    target_compile_definitions(${SPH_VIS} PRIVATE -DGL_SILENCE_DEPRECATION)
endif()

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
set(SPH_VIS_INSTALL_COMPONENT ${SPH_VIS}_install)
set(SPH_VIS_INSTALL_FOLDER visualization)

install(TARGETS ${SPH_VIS}
        RUNTIME DESTINATION ${SPH_VIS_INSTALL_FOLDER}
        COMPONENT ${SPH_VIS_INSTALL_COMPONENT}
)

install(FILES ${SPH_VIS_RESOURCES}
        DESTINATION ${SPH_VIS_INSTALL_FOLDER}
        COMPONENT ${SPH_VIS_INSTALL_COMPONENT}
)

install_dependencies(${SPH_VIS} ${SPH_VIS_INSTALL_COMPONENT} ${SPH_VIS_INSTALL_FOLDER})

add_custom_target(install_visualization
    COMMAND ${CMAKE_COMMAND} 
        --install ${CMAKE_CURRENT_BINARY_DIR} 
        --config $<CONFIGURATION> 
        --component ${SPH_VIS_INSTALL_COMPONENT}
)

add_dependencies(install_visualization ${SPH_VIS})
