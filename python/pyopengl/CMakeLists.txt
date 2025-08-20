cmake_minimum_required(VERSION 3.4...3.18)
project(pyopenpgl)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

set(PYOPENPGL_OPTIONS -msse2 -msse4.1 -O3 -Wall -ffp-model=precise -Wformat -Wformat-security -fvisibility=hidden -fvisibility-inlines-hidden -fno-strict-aliasing -fno-tree-vectorize)

add_subdirectory(../ext/pybind11 build)

pybind11_add_module(pyopenpgl
    python.cpp
)

set_source_files_properties(python.cpp PROPERTIES COMPILE_FLAGS ${PYOPENPGL_OPTIONS})

#target_include_directories(libguiding PUBLIC include)
target_include_directories(pyopenpgl PRIVATE ${CMAKE_INSTALL_PREFIX}/include)

message(STATUS "openpgl_DIR = ${openpgl_DIR}")



if (NOT ${OPENPGL_TBB_ROOT} STREQUAL "")
    set(TBB_FIND_PACKAGE_OPTION "NO_DEFAULT_PATH")
    set(TBB_ROOT ${OPENPGL_TBB_ROOT})
    list(APPEND CMAKE_PREFIX_PATH ${OPENPGL_TBB_ROOT})
endif()

#find_package(TBB REQUIRED)
#find_package(openpgl REQUIRED)

message(STATUS "TBB_INCLUDE_DIRS = ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB_FOUND = ${TBB_FOUND}")

target_link_libraries(pyopenpgl PRIVATE openpgl)
target_link_libraries(pyopenpgl PRIVATE TBB::tbb)

set (PYOPENPGL_RPATH
  "$ORIGIN:$ORIGIN/..")

set_target_properties(pyopenpgl PROPERTIES
  PREFIX ""
  INSTALL_RPATH ${PYOPENPGL_RPATH}
  )

install(TARGETS pyopenpgl DESTINATION lib/python)


pybind11_add_module(pyopenpgldebug
    pyopenpgldebug.cpp
)

set_source_files_properties(pyopenpgldebug.cpp PROPERTIES COMPILE_FLAGS ${PYOPENPGL_OPTIONS})
#target_include_directories(libguiding PUBLIC include)

message(STATUS "TEST=${CMAKE_CURRENT_SOURCE_DIR}/../../")

target_include_directories(pyopenpgldebug PRIVATE ${CMAKE_INSTALL_PREFIX}/include)
target_include_directories(pyopenpgldebug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_include_directories(pyopenpgldebug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third-party)

target_link_libraries(pyopenpgldebug PRIVATE openpgl)
target_link_libraries(pyopenpgldebug PRIVATE TBB::tbb)

set (PYOPENPGLDEBUG_RPATH
  "$ORIGIN:$ORIGIN/..")

set_target_properties(pyopenpgldebug PROPERTIES
  PREFIX ""
  INSTALL_RPATH ${PYOPENPGLDEBUG_RPATH}
  )

install(TARGETS pyopenpgldebug DESTINATION lib/python)