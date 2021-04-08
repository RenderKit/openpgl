## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0


add_library(${PROJECT_NAME}
  #${RKCOMMON_RESOURCE}
  api/api.cpp
  ../third-party/embree/common/simd/sse.cpp
  #openpgl.cpp
)

set (CMAKE_CXX_FLAGS "-msse2")
#add_compile_options()

FIND_PACKAGE(embree 3.0 REQUIRED)
get_target_property(embree_include_dir embree INTERFACE_INCLUDE_DIRECTORIES)


target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>
    ${embree_include_dir}
    #$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/>
    #$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)