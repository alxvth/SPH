vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO facebookresearch/faiss
    REF "v${VERSION}"
    SHA512 64d333e3cf561a65a9dcb78bb04f76073047b1149ce4778e4d65aa809928bedbd43b2b0a3362e8336664feae3d09167702ef68abddce3c86bc70cdb9551bc65c
    HEAD_REF master
    PATCHES
        msvc-template.diff
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        "avx2" FAISS_ENABLE_AVX2
)

set(CMAKE_OPTIONS_LIST
    -DFAISS_ENABLE_EXTRAS=OFF
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_MKL=OFF
    -DFAISS_ENABLE_PYTHON=OFF
    -DBUILD_TESTING=OFF
    -DMKL_FOUND=OFF
)

set(FAISS_OPT_LEVEL "generic")

if(${FAISS_ENABLE_AVX2} AND NOT VCPKG_TARGET_ARCHITECTURE STREQUAL "arm" AND NOT VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    message(STATUS "Set AVX2: ON")
    set(FAISS_OPT_LEVEL "avx2")
endif()

list(APPEND CMAKE_OPTIONS_LIST "-DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL}")

if(VCPKG_TARGET_IS_WINDOWS)
    message(STATUS "Set OpenMP_RUNTIME_MSVC: llvm")
    list(APPEND CMAKE_OPTIONS_LIST "-DOpenMP_RUNTIME_MSVC=llvm")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${CMAKE_OPTIONS_LIST}
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
