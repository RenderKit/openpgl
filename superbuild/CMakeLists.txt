## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## Global settings ##

cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
)

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_BINARY_DIR}/install"
      CACHE STRING "Final install location." FORCE)
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

## ======================================================================== ##
## Project setup.
## ======================================================================== ##

project(OpenPGL_superbuild)

include(ExternalProject)
include(GNUInstallDirs)
include(ProcessorCount)

## Setup top-level variables ##

set(installDir ${CMAKE_INSTALL_PREFIX})

get_filename_component(INSTALL_DIR_ABSOLUTE
  ${installDir} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

ProcessorCount(PROCESSOR_COUNT)

if(NOT PROCESSOR_COUNT EQUAL 0)
  set(BUILD_JOBS ${PROCESSOR_COUNT} CACHE STRING "Number of build jobs '-j <n>'")
else()
  set(BUILD_JOBS 4 CACHE STRING "Number of build jobs '-j <n>'")
endif()

if (${CMAKE_VERSION} VERSION_GREATER 3.11.4)
  set(PARALLEL_JOBS_OPTS -j ${BUILD_JOBS})
endif()

set(DEFAULT_BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release ${PARALLEL_JOBS_OPTS})

# We must replace separators so that ExternalProject_Add can work with the
# paths we generate.
macro(add_to_prefix_path PATH)
  list(APPEND CMAKE_PREFIX_PATH ${PATH})
  string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endmacro()

## ======================================================================== ##
## Superbuild options.
## ======================================================================== ##

option(INSTALL_IN_SEPARATE_DIRECTORIES
  "Install libraries into their own directories under CMAKE_INSTALL_PREFIX"
  OFF
)

option(BUILD_DEPENDENCIES_ONLY
  "Do not build Open PGL itself, only its dependencies" 
  OFF
)

## ======================================================================== ##
## Build dependencies.
##
## Use the BUILD_* options to enable or disable building in superbuild. If 
## These options are OFF, then required dependencies must be available in the 
## environment.
##
## For most of the dependencies, you may override the *_URL and *_HASH
## variables to select different versions.
## ======================================================================== ##

option(BUILD_ISPC "Build the Intel SPMD Program Compiler or search in environment?" ON)
if (BUILD_ISPC)
  set(ISPC_VERSION "1.15.0")
  set(ISPC_BASE_URL "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}")
  if (APPLE)
    set(_ISPC_URL "${ISPC_BASE_URL}/ispc-v${ISPC_VERSION}-macOS.tar.gz")
    set(_ISPC_HASH "eb736abb5ed197504113aa88e5ea04ca13f47add87a96b386be95b14776b2048")
  elseif(WIN32)
    set(_ISPC_URL "${ISPC_BASE_URL}/ispc-v${ISPC_VERSION}-windows.zip")
    set(_ISPC_HASH "76a10ba670b5163c22dfd354d003535fd0cf0cfd1287aed8724e6cca0f2105b7")
  else()
    set(_ISPC_URL "${ISPC_BASE_URL}/ispc-v${ISPC_VERSION}-linux.tar.gz")
    set(_ISPC_HASH "b67f50ab16b38d29e28b0a2dbb9733fd6fc1276cb5a5be0cac78e356941f881f")
  endif()
  set(ISPC_URL "${_ISPC_URL}" CACHE STRING "URL of the ISPC archive.")
  set(ISPC_HASH "${_ISPC_HASH}" CACHE STRING "SHA256 hash of the ISPC archive.")
  include(dep_ispc)
endif()


option(BUILD_TBB "Build Intel Threading Building Blocks or search in environment?" ON)
if (BUILD_TBB)
  set(TBB_VERSION "2020.3" CACHE STRING "TBB version to download")
  if (TBB_VERSION VERSION_LESS 2021)
    set(TBB_BASE_URL "https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}/tbb-${TBB_VERSION}")
  else()
    set(TBB_BASE_URL "https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}/oneapi-tbb-${TBB_VERSION}")
  endif()
  if (APPLE)
    set(_TBB_URL "${TBB_BASE_URL}-mac.tgz")
    set(_TBB_HASH "456d1319e7418c761090e22be365c88e521ab11e0630250bc4addb0e6c8e1911")
    set(TBB_LIB_SUBDIR "")
  elseif(WIN32)
    set(_TBB_URL "${TBB_BASE_URL}-win.zip")
    set(_TBB_HASH "cda37eed5209746a79c88a658f8c1bf3782f58bd9f9f6ba0da3a16624a9bfaa1")
    set(TBB_LIB_SUBDIR "intel64/vc14")
  else()
    set(_TBB_URL "${TBB_BASE_URL}-lin.tgz")
    set(_TBB_HASH "bb8cddd0277605d3ee7f4e19b138c983f298d69fcbb585385b59ef7239d5ef83")
    set(TBB_LIB_SUBDIR "intel64/gcc4.8")
  endif()
  set(TBB_URL "${_TBB_URL}" CACHE STRING "URL of the TBB archive.")
  set(TBB_HASH "${_TBB_HASH}" CACHE STRING "SHA256 hash of the TBB archive.")
  include(dep_tbb)
endif()

# Embree can either be build from source or downloaded as binaries.
option(BUILD_EMBREE "Build Intel Embree or search in environment?" ON)
option(BUILD_EMBREE_FROM_SOURCE "Build Embree from source or use pre-built version? (Only used when BUILD_EMBREE=ON)" OFF)
if (BUILD_EMBREE)
  set(EMBREE_VERSION "v3.12.2" CACHE STRING "Embree version to download")
  if (BUILD_EMBREE_FROM_SOURCE)
    set(_EMBREE_URL "https://github.com/embree/embree/archive/${EMBREE_VERSION}.zip")
    set(_EMBREE_HASH "83aa90b1526fed8db80e800bb5c080cce3f06a8729b6dc24e597bac7f931cf98")
  else()
    # Embree binary package URLs do not use the "v" prefix
    string(REPLACE "v" "" EMBREE_VERSION_NUMBER ${EMBREE_VERSION})
    set(EMBREE_BASE_URL "https://github.com/embree/embree/releases/download/${EMBREE_VERSION}")
    if (APPLE)
      set(_EMBREE_URL "${EMBREE_BASE_URL}/embree-${EMBREE_VERSION_NUMBER}.x86_64.macosx.zip")
      set(_EMBREE_HASH "076debd5a734ae5348791eafe21aefbbba7fbc5f32cf6e9b2ae55b3a23d5ca0b")
    elseif (WIN32)
      set(_EMBREE_URL "${EMBREE_BASE_URL}/embree-${EMBREE_VERSION_NUMBER}.x64.vc14.windows.zip")
      set(_EMBREE_HASH "fa495acc3c1494c9affebd0a9a707d462ca2fa9d303acc230caf942434bc6bcc")
    else()
      set(_EMBREE_URL "${EMBREE_BASE_URL}/embree-${EMBREE_VERSION_NUMBER}.x86_64.linux.tar.gz")
      set(_EMBREE_HASH "0cb88651ca08798eb6d3d560bb2adc568973d3b74f9acdcbe756b7d9296b6fc1")
    endif()
  endif()
  set(EMBREE_URL "${_EMBREE_URL}"  CACHE STRING "URL of the Embree source archive.")
  set(EMBREE_HASH "${_EMBREE_HASH}" CACHE STRING "SHA256 hash for the Embree source archive.")
  include(dep_embree)
endif()

## ======================================================================== ##
## Build OpenPGL
## ======================================================================== ##

set(COMPONENT_NAME openpgl)

set(COMPONENT_PATH ${INSTALL_DIR_ABSOLUTE})
if (INSTALL_IN_SEPARATE_DIRECTORIES)
  set(COMPONENT_PATH ${INSTALL_DIR_ABSOLUTE}/${COMPONENT_NAME})
endif()

if (NOT BUILD_DEPENDENCIES_ONLY)

  set(OPENPGL_ARGS
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
    -DBUILD_PYTHON=OFF
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${COMPONENT_PATH}
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DCMAKE_INSTALL_DOCDIR=${CMAKE_INSTALL_DOCDIR}
    -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
  )

  if(BUILD_STATIC)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DBUILD_STATIC=${BUILD_STATIC}
    )
  endif()

  if(TBB_ROOT)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DTBB_ROOT=${TBB_ROOT}
    )
  endif()

  ExternalProject_Add(openpgl
    PREFIX openpgl
    DOWNLOAD_COMMAND ""
    STAMP_DIR openpgl/stamp
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
    BINARY_DIR openpgl/build
    LIST_SEPARATOR | # CMAKE_PREFIX_PATH uses alternate separator.
    CMAKE_ARGS
      ${OPENPGL_ARGS}
      #$<$<BOOL:${BUILD_ISPC}>:-DISPC_EXECUTABLE=${ISPC_PATH}>
    BUILD_COMMAND ${DEFAULT_BUILD_COMMAND}
    BUILD_ALWAYS OFF
  )

  ExternalProject_Add_StepDependencies(openpgl
  configure
    $<$<BOOL:${BUILD_ISPC}>:ispc>
    $<$<BOOL:${BUILD_TBB}>:tbb>
    $<$<BOOL:${BUILD_EMBREE}>:embree>
  )
endif()
