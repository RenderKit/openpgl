cmake_minimum_required(VERSION 3.1)
project(pyopenpgl VERSION 0.3.0)

#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -msse2 -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer ")
set(CMAKE_VERBOSE_MAKEFILE ON)

pybind11_add_module(pyopenpgl32 pyopenpgl.cpp ../third-party/embreeSrc/common/simd/sse.cpp)

target_compile_definitions(pyopenpgl32 PUBLIC -DPYGUIDE_MAX_COMPONENTS=32)
target_include_directories(pyopenpgl32
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)


pybind11_add_module(pyopenpgl64 pyopenpgl.cpp ../third-party/embreeSrc/common/simd/sse.cpp)

target_compile_definitions(pyopenpgl64 PUBLIC -DPYGUIDE_MAX_COMPONENTS=64)
target_include_directories(pyopenpgl64
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)


pybind11_add_module(pyopenpgl128 pyopenpgl.cpp ../third-party/embreeSrc/common/simd/sse.cpp)

target_compile_definitions(pyopenpgl128 PUBLIC -DPYGUIDE_MAX_COMPONENTS=128)
target_include_directories(pyopenpgl128
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

