# CMake file for Heat Equation Python Driver Example
#
# This demonstrates the minimal setup needed for a Python interface
# to AMReX C++ simulation code.
#
# Step 1: Create and enter a build directory:
#
#   mkdir build
#   cd build
#
# Step 2: Configure with CMake from inside the build directory:
#
#   cmake ..
#
# Step 3: Build the configuration:
#
#   cmake --build . -j4
#
# Step 4: Test the Python interface:
#
#   python ../test.py

cmake_minimum_required(VERSION 3.16)

# Project name and source file languages
project(HeatEquation_PythonDriver
        LANGUAGES C CXX)

# Use pyamrex pybind11 infrastructure
include(pybind11.cmake)

# Add the main executable
add_executable(HeatEquation_PythonDriver main.cpp)

# Add the pybind11 module including main simulation logic
pybind11_add_module(amrex_heat bindings.cpp main.cpp)

# To use a pre-installed AMReX build, run:
#    cmake -DAMReX_ROOT=/path/to/installdir
# Otherwise cmake will download AMReX from GitHub

if(NOT DEFINED AMReX_ROOT)
  message("-- Download and configure AMReX from GitHub")

  #Download AMReX from GitHub
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)

  # Force position-independent code for shared libraries
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  FetchContent_Declare(
    amrex_code
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git/
    GIT_TAG        origin/development
    )

  FetchContent_Populate(amrex_code)

  # CMake will read the files in these directories to configure, build
  # and install AMReX.
  add_subdirectory(${amrex_code_SOURCE_DIR} ${amrex_code_BINARY_DIR})

else()

  # Add AMReX install
  message("-- Searching for AMReX install directory at ${AMReX_ROOT}")
  find_package(AMReX PATHS ${AMReX_ROOT}/lib/cmake/AMReX/AMReXConfig.cmake)

endif()

# Link AMReX to both targets
target_link_libraries(HeatEquation_PythonDriver PRIVATE AMReX::amrex)
target_link_libraries(amrex_heat PRIVATE AMReX::amrex)

## Copy input files to build directory
file( COPY ${CMAKE_SOURCE_DIR}/inputs DESTINATION . )