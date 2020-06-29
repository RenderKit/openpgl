## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## Global CMake options ##

cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

include(GNUInstallDirs)

## Establish project ##

project(rkguide VERSION 0.0.1 LANGUAGES CXX)

## Build library and  ##
add_subdirectory(rkguide)

add_subdirectory(python)