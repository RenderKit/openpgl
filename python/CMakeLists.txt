cmake_minimum_required(VERSION 2.8.12)
project(pyrkguide)

add_subdirectory(pybind11)


pybind11_add_module(pyrkguide pyrkguide.cpp)

target_include_directories(pyrkguide
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)