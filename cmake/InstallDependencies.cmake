# Call as e.g.
# install_dependencies(${SPH_EVAL} ${SPH_EVAL}_install evaluation ${SPH_LIB})
# with:
#   main_target: ${SPH_EVAL}
#   install_component: ${SPH_EVAL}_install
#   install_location: evaluation (is relative to ${CMAKE_INSTALL_PREFIX})
#   <extra arguments>: other targets to consider during dependency resolution (here: ${SPH_LIB})
function(install_dependencies main_target install_component install_location)
    
    set(MAIN_PROJECT_TARGET ${main_target})
    set(INSTALL_COMPONENT_TARGET ${install_component})
    set(INSTALL_LOCATION_TARGET ${install_location})

    list(APPEND DEPENDENCIES_FOLDERS "$<TARGET_FILE_DIR:${main_target}>")
    foreach(TARGET_STR IN LISTS ARGN)
        list(APPEND DEPENDENCIES_FOLDERS "$<TARGET_FILE_DIR:${TARGET_STR}>")
    endforeach()

    include(CMakePackageConfigHelpers)
    set(CONFIG_FILE_IN "${SPH_CMAKE_MODULES_PATH}/InstallDependencies.cmake.in")
    set(CONFIG_FILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/InstallDependencies_${main_target}.cmake")
    configure_file("${CONFIG_FILE_IN}" "${CONFIG_FILE_OUT}" @ONLY)

    include("${CONFIG_FILE_OUT}")
endfunction()
