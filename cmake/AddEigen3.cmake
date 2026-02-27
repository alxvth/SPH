# Currently, this file is NOT used
set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Toggles the building of the Eigen demos" FORCE)
fetch_content_url(eigen "https://gitlab.com/libeigen/eigen/-/archive/e67c494cba7180066e73b9f6234d0b2129f1cdf5/eigen-e67c494cba7180066e73b9f6234d0b2129f1cdf5.tar.gz") # 3.4 as of 14/11/24

function(link_eigen TARGET_NAME TARGET_SCOPE)

    target_link_libraries(${TARGET_NAME} ${TARGET_SCOPE} Eigen3::Eigen)

    if(NOT MSVC)
        find_dependency(BLAS REQUIRED)
        find_dependency(LAPACK REQUIRED)
        target_link_libraries(${TARGET_NAME} PRIVATE BLAS::BLAS)
        target_link_libraries(${TARGET_NAME} PRIVATE LAPACK::LAPACK)

        target_compile_definitions(${TARGET_NAME} ${TARGET_SCOPE} 
            EIGEN_USE_BLAS 
            EIGEN_USE_LAPACKE_STRICT
            lapack_complex_float=std::complex<float>
            lapack_complex_double=std::complex<double>
            )
    else()
        target_compile_options(${TARGET_NAME} ${TARGET_SCOPE} /bigobj)
    endif()

endfunction()
