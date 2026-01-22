#
# Copyright (c) 2023-24 Armin Sobhani
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
cmake_minimum_required(VERSION 3.25.2...3.31)

## detect if gynx is being bundled, disable unit tests in that case
#
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else()
  set(NOT_SUBPROJECT OFF)
endif()

## life is about choices...
#
option(GYNX_ENABLE_TESTS "Enable the unit tests ?" ON)
option(GYNX_ENABLE_BENCHMARKS "Enable benchmarks ?" ON)

## finally our project...
#
project(
  gynx
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Biological sequences for Modern C++"
)

## prevent in-source builds
#
if (CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "Building in-source is not supported! Create a build dir and remove ${CMAKE_SOURCE_DIR}/CMakeCache.txt")
endif()

## necessary modules
#
include(GNUInstallDirs)
include(FetchContent)
include(CheckLanguage)

## check for compiler support for OpenMP
#
find_package(OpenMP REQUIRED)

## check for CUDA support
#
check_language(CUDA 12)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

## check for ROCm support
#
check_language(HIP 6)
if (CMAKE_HIP_COMPILER)
  enable_language(HIP)
  find_package(hip HINTS $ENV{HIP_PATH} /opt/rocm/hip /opt/rocm)
endif()

## check for zlib
#
find_package(ZLIB REQUIRED)

## check for ranx
#
find_package(ranx CONFIG COMPONENTS openmp
  HINTS $ENV{HOME} $ENV{HOME}/.local /usr/local /opt/local /opt
)
if(NOT ranx_FOUND)
  message(STATUS "Fetching ranx library...")
  FetchContent_Declare(
    ranx
    GIT_REPOSITORY https://github.com/arminms/ranx.git
    GIT_TAG main
  )
  # setting required ranx components
  set(RANX_COMPONENTS openmp CACHE STRING "Required components")
  FetchContent_MakeAvailable(ranx)
endif()

## check for g3p
#
find_package(
  g3p CONFIG
  HINTS $ENV{HOME} $ENV{HOME}/.local /usr/local /opt/local /opt
)
if(NOT g3p_FOUND)
    message(STATUS "Fetching g3p library...")
    FetchContent_Declare(
      g3p
      GIT_REPOSITORY https://github.com/arminms/g3p.git
      GIT_TAG        v1.2.0
    )
    FetchContent_MakeAvailable(g3p)
endif()

## defining gynx target
#
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
target_link_libraries(${PROJECT_NAME} INTERFACE
  OpenMP::OpenMP_CXX
  ZLIB::ZLIB
  g3p::g3p
)
if(MSVC)
  target_compile_options(${PROJECT_NAME} INTERFACE /openmp:llvm -openmp:experimental)
endif()
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${gynx_VERSION}
  SOVERSION ${gynx_VERSION_MAJOR}
)

## specifying the export name for the targets
#
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
)

## installing gynx header files
#
add_subdirectory(include)

## installing gynx export target file
#
install(
  EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

## generate the config file that includes the exports
#
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

## creating version file
#
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION "${gynx_VERSION_MAJOR}.${gynx_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

## installing gynx export config files
#
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

## make export target available from the build tree
#
export(EXPORT ${PROJECT_NAME}-targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-targets.cmake"
  NAMESPACE ${PROJECT_NAME}::
)

## include unit tests only when gynx is not being used as a subproject
#
if (NOT_SUBPROJECT)

  ## add unit tests
  #
  if(${GYNX_ENABLE_TESTS})
    enable_testing()
    find_package(
      Catch2 3
      HINTS $ENV{HOME} /usr/local /opt/local /opt
    )
    if(NOT Catch2_FOUND)
        message(STATUS "Fetching Catch2 library...")
        FetchContent_Declare(
          catch2
          GIT_REPOSITORY https://github.com/catchorg/Catch2.git
          GIT_TAG        v3.5.2
        )
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    endif()
    include(Catch)
    add_subdirectory(test)
  endif()

endif(NOT_SUBPROJECT)

## include benchmarks only when gynx is not being used as a subproject
#
if (NOT_SUBPROJECT)

  ## add benchmarks
  #
  if(${GYNX_ENABLE_BENCHMARKS})
    find_package(benchmark
      CONFIG
      HINTS $ENV{HOME} /usr/local /opt/local /opt
    )
    if(NOT benchmark_FOUND)
      message(STATUS "Fetching Google Benchmark library...")
      FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.7.1
      )
      set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
        "Enable testing of the benchmark library."
      )
      FetchContent_MakeAvailable(benchmark)
    endif()
    add_subdirectory(perf)
  endif()

endif(NOT_SUBPROJECT)
