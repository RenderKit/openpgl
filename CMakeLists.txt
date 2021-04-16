## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## Global CMake options ##

cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

## Establish project ##

project(openpgl VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(openpgl_macros)

## openpgl specific configuration ##

option(BUILD_PYTHON "Build python wrapper 'pyopenpgl'."  OFF)

find_package(embree 3.6.1 REQUIRED)

openpgl_create_embree_target()

## Build library and  ##
add_subdirectory(openpgl)

if (BUILD_PYTHON)
    add_subdirectory(python)
endif()