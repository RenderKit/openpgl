## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0


add_library(${PROJECT_NAME} SHARED
  #${RKCOMMON_RESOURCE}
  api/api.cpp
  ../third-party/embreeSrc/common/simd/sse.cpp
  #openpgl.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -msse2 -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded -fopenmp -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer ")

#add_compile_options()

FIND_PACKAGE(embree 3.6 REQUIRED)
get_target_property(embree_include_dir embree INTERFACE_INCLUDE_DIRECTORIES)


message(STATUS "CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

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


## Configure OpenPGL installation ##

openpgl_install_library(${PROJECT_NAME})

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/include/${PROJECT_NAME}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  #PATTERN "*.isph"
)

install(FILES
  ${PROJECT_SOURCE_DIR}/LICENSE.txt
#  ${PROJECT_SOURCE_DIR}/third-party-programs.txt
#  ${PROJECT_SOURCE_DIR}/third-party-programs-Embree.txt
#  ${PROJECT_SOURCE_DIR}/third-party-programs-OSPRay.txt
#  ${PROJECT_SOURCE_DIR}/third-party-programs-TBB.txt
  ${PROJECT_SOURCE_DIR}/README.md
  ${PROJECT_SOURCE_DIR}/CHANGELOG.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/${PROJECT_NAME})

## Version header ##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h
  @ONLY
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}>
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)