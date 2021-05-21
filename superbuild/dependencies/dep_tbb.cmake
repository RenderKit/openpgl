## Copyright 2019-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(COMPONENT_NAME tbb)

set(COMPONENT_PATH ${INSTALL_DIR_ABSOLUTE})
if (INSTALL_IN_SEPARATE_DIRECTORIES)
  set(COMPONENT_PATH ${INSTALL_DIR_ABSOLUTE}/${COMPONENT_NAME})
endif()

set(TBB_HASH_ARGS "")
if (NOT "${TBB_HASH}" STREQUAL "")
  set(TBB_HASH_ARGS URL_HASH SHA256=${TBB_HASH})
endif()

# handle changed paths in TBB 2021
if (TBB_VERSION VERSION_LESS 2021)
  set(TBB_SOURCE_INCLUDE_DIR tbb/include)
  set(TBB_SOURCE_LIB_DIR tbb/lib/${TBB_LIB_SUBDIR})
  set(TBB_SOURCE_BIN_DIR tbb/bin/${TBB_LIB_SUBDIR})
else()
  set(TBB_SOURCE_INCLUDE_DIR include)
  set(TBB_SOURCE_LIB_DIR lib/${TBB_LIB_SUBDIR})
  set(TBB_SOURCE_BIN_DIR redist/${TBB_LIB_SUBDIR})
endif()

ExternalProject_Add(${COMPONENT_NAME}
  PREFIX ${COMPONENT_NAME}
  DOWNLOAD_DIR ${COMPONENT_NAME}
  STAMP_DIR ${COMPONENT_NAME}/stamp
  SOURCE_DIR ${COMPONENT_NAME}/src
  BINARY_DIR ${COMPONENT_NAME}
  URL ${TBB_URL}
  ${TBB_HASH_ARGS}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy_directory
    <SOURCE_DIR>/${TBB_SOURCE_INCLUDE_DIR}
    ${COMPONENT_PATH}/include
  BUILD_ALWAYS OFF
)

# We copy the libraries into the main lib dir. This makes it easier
# to set the correct library path.
ExternalProject_Add_Step(${COMPONENT_NAME} install_lib
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
  <SOURCE_DIR>/${TBB_SOURCE_LIB_DIR} ${COMPONENT_PATH}/lib
  DEPENDEES install
)

if (WIN32)
  # DLLs on Windows are in the bin subdirectory.
  ExternalProject_Add_Step(${COMPONENT_NAME} install_dll
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
    <SOURCE_DIR>/${TBB_SOURCE_BIN_DIR} ${COMPONENT_PATH}/bin
    DEPENDEES install_lib
  )
endif()

set(TBB_PATH ${COMPONENT_PATH})

add_to_prefix_path(${COMPONENT_PATH})
