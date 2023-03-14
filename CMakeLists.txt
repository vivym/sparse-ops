
cmake_minimum_required(VERSION 3.17...3.25)

project(
  SparseOps
  VERSION 0.1
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pytorch
find_package(Torch REQUIRED)
set(CMAKE_CXXFLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED PATHS "${TORCH_INSTALL_PREFIX}/lib")

# cuda
find_package(CUDAToolkit REQUIRED)

include(cmake/CPM.cmake)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

add_subdirectory(src/sparse_ops)
