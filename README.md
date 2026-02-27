# Superpixel Hierarchies and Embeddings for High-Dimensional Images
Create superpixel hierarchies for high-dimensional images and compute embeddings of each hierarchy level.

```bash
git clone git@github.com:alxvth/SPH.git
```

## Usage

Basic:
```cpp
#include <sph/ComputeHierarchy.hpp>
#include <sph/ComputeEmbedding.hpp>

const std::filesystem::path loadDirectory = ""; // contains single-page tiffs corresponding to each image channel
sph::utils::ImageStack myImage = sph::utils::loadTiffImageStack(loadDirectory);

sph::ComputeHierarchy spatialHierarchy;
spatialHierarchy.setData(myImage);
spatialHierarchy.compute();

const auto& hierarchy = spatialHierarchy.getImageHierarchy()->getHierarchy();

// Compute embeddings of the first abstraction level
int64_t level = 1;
sph::ComputeEmbedding embedder;
embedder.computeTSNE(spatialHierarchy.getLevelSimilarities()->getProbDist(level));
std::vector<float> tSNE = embedder.getEmbedding();
```

Advanced (specify non-default settings):
```cpp
auto nns = sph::NearestNeighborsSettings();
auto ihs = sph::ImageHierarchySettings();
auto lss = sph::LevelSimilaritiesSettings();
auto rws = sph::utils::RandomWalkSettings();

ihs.componentSim = utils::ComponentSim::GEO_CENTROID;

sph::ComputeHierarchy spatialHierarchy;
spatialHierarchy.init(myImage, ihs, lss, rws, nns);
spatialHierarchy.compute();

// Compute embeddings of the first abstraction level
int64_t level = 1;
auto ems = sph::ComputeEmbeddingSettings();
ems.tsne.numIterations = 2000;

sph::ComputeEmbedding embedder;
embedder.setSettings(ems);

embedder.computeTSNE(spatialHierarchy.getLevelSimilarities()->getProbDist(level));
std::vector<float> tSNE = embedder.getEmbedding();
```

## Building
This projects uses [vcpkg](https://github.com/microsoft/vcpkg/) to install dependencies like [Boost Graph](https://github.com/boostorg/graph). The [vcpkg.json](./vcpkg.json) manifest file can be used together with CMake to automate this process. 
After setting up vcpkg, you can incorporate vcpkg with CMake via the command line
```bash
> cmake -B [build directory] -S . "-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake"
> cmake --build [build directory] 
```
or in the CMake GUI via "Specify toolchain for cross-compiling", or in your IDE's specific fashion.

When using presets, ensure to set the environment variable `VCPKG_ROOT` to the vcpkg install location, preferably a short path like `C:\vcpkg` (Windows) or `$HOME/vcpkg` (Linux).

Setup vcpkg, on Linux:
```bash
cd $HOME
sudo apt update
sudo apt install -y build-essential gfortran pkg-config zip unzip
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
echo -e '\n# Vcpk\nexport VCPKG_ROOT=$HOME/vcpkg' >> ~/.bashrc
source ~/.bashrc
```

You need at least [CMake](https://gitlab.kitware.com/cmake/cmake) 3.31. Your OS package manager might not provide this version yet.

Setup CMake, on Linux:
```bash
cd $HOME
sudo apt purge cmake
sudo apt install -y libssl-dev
mkdir cmake && cd cmake
mkdir cmake_install
git clone https://gitlab.kitware.com/cmake/cmake.git
mv cmake cmake_source && cd cmake_source
git checkout v3.31.11
./bootstrap --prefix=$HOME/cmake/cmake_install
make -j$(nproc) && make install
echo -e '\n# Custom cmake\nexport PATH=$HOME/cmake/cmake_install/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### On Windows

It might be handy to have a Fortran compiler, [Intel® Fortran Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler-download.html):
```powershell
winget install --id=Intel.FortranCompiler  -e
```

Consider setting `CMAKE_Fortran_COMPILER` to `C:/Program Files (x86)/Intel/oneAPI/compiler/2024.2/bin/ifx.exe` for vcpkg to make use of the compiler. Alternatively, add this path as `FC` to your environment paths.

### On Linux
```bash
sudo apt install -y build-essential ninja-build gfortran libssl-dev libtbb-dev libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
cmake --preset ninja-multi
cmake --build --preset release-ninja --target SPHEvaluation
```
or with any other target specification.

### WSL Notes
To make use of hardware acceleration for the GPU t-SNE implementation, you might need to adjust some driver settings:
```bash
sudo apt install -y ppa-purge mesa-utils
sudo ppa-purge ppa:kisak/kisak-mesa
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA glxinfo -B
```
See [this issue](https://github.com/microsoft/WSL/issues/12412#issuecomment-2569495986) and [the docs](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg).

## Visualization tool
A simple OpenGL based tool for testing some library functions is build with the CMake variable `SPH_BUILD_VIS=ON`.

## Testing
This library sets up some `./tests` with [catch2](https://github.com/catchorg/Catch2). Enable building tests with the CMake variable `SPH_BUILD_TESTS=ON`.

## Evaluation
Current "best" practice is to copy `eval_settings.json` to `eval_settings_local.json` and use that as input for the `eval` executable:
```powershell
.out/install/evaluation/SPHEvaluation/exe ./eval_settings_local.json
```

When using Visual Studio you should add the full path of your evaluation settings file to `SPHEvaluation > Properties > Debugging > Command Arguments` for all build configurations that you are interested in.

Enable building tests with the CMake variable `SPH_BUILD_EVAL=ON`.

## Comparison libraries and data
See [the comparison README](./comparison/README.md) for more info on setting up comparison libraries and evaluation data.

Either clone this repo with submodules:
```bash
git clone --recurse-submodules git@github.com:alxvth/SPH.git
```
or initialize the submodules after cloning:
```bash
git submodule update --init --recursive
```

## Third party libraries
In addition to the third party libraries installed from [vcpkg.json](./vcpkg.json), CMake will fetch [HDILib](https://github.com/biovault/HDILib) during configuration.
