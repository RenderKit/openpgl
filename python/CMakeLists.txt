cmake_minimum_required(VERSION 2.8.12)
project(pyrkguide)

add_subdirectory(pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -msse2 -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpadded -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer ")
set(CMAKE_VERBOSE_MAKEFILE ON)

pybind11_add_module(pyrkguide pyrkguide.cpp)

target_include_directories(pyrkguide
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)