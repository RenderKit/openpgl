## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

ADD_LIBRARY(imgui STATIC
  ../third-party/imgui/imgui.cpp
  ../third-party/imgui/imgui_draw.cpp
  ../third-party/imgui/imgui_widgets.cpp
  ../third-party/imgui/imgui_tables.cpp
  ../third-party/imgui/backends/imgui_impl_glfw.cpp
  ../third-party/imgui/backends/imgui_impl_opengl3.cpp
)

TARGET_INCLUDE_DIRECTORIES(imgui PUBLIC ../third-party/imgui/ ../third-party/imgui/backends)
TARGET_LINK_LIBRARIES(imgui glfw GLEW)
#SET_PROPERTY(TARGET imgui PROPERTY FOLDER tutorials/common)
SET_PROPERTY(TARGET imgui APPEND PROPERTY COMPILE_FLAGS " ${FLAGS_LOWEST}")

TARGET_COMPILE_DEFINITIONS(imgui PUBLIC IMGUI_DISABLE_SSE)