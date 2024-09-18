cmake_minimum_required(VERSION 3.15)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

#add_subdirectory(third-party/glm)
add_subdirectory(imgui)

#find_package(imgui REQUIRED)
find_package(glfw3 REQUIRED)
#find_package(glew REQUIRED)

#find_package (openpgl REQUIRED)

FIND_PACKAGE(TBB REQUIRED)
include_directories(TBB::tbb)


######################
# soac

add_executable (sc src/cmd/sc.cpp)
add_executable (openpgl::sc ALIAS sc)

#target_compile_definitions (sc PRIVATE ${PBRT_DEFINITIONS})
#target_compile_options (sc PUBLIC ${PBRT_CXX_FLAGS})
#target_link_libraries (sc PRIVATE pbrt_warnings pbrt_opt)

set_target_properties (sc PROPERTIES OUTPUT_NAME sc)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/caches_sc.h
    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/caches.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/caches.gs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/caches.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/caches_sc.h
    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/caches.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/caches.gs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/caches.fs)
set (OPENPGL_SC_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/shaders/caches_sc.h)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/cachesBBoxes_sc.h
    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/cachesBBoxes.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/cachesBBoxes.gs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/cachesBBoxes.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/cachesBBoxes_sc.h
    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/cachesBBoxes.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/cachesBBoxes.gs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/cachesBBoxes.fs)
set (OPENPGL_SC_GENERATED ${OPENPGL_SC_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/shaders/cachesBBoxes_sc.h)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/sampledistribution_sc.h
    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/sampledistribution.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/sampledistribution.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/sampledistribution_sc.h
    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/sampledistribution.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/sampledistribution.fs)
set (OPENPGL_SC_GENERATED ${OPENPGL_SC_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/shaders/sampledistribution_sc.h)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/coloredsamples_sc.h
    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/coloredsamples.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/coloredsamples.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/coloredsamples_sc.h
    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/coloredsamples.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/coloredsamples.fs)
set (OPENPGL_SC_GENERATED ${OPENPGL_SC_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/shaders/coloredsamples_sc.h)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/framebuffer_sc.h
    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/framebuffer.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/framebuffer.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/framebuffer_sc.h
    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/framebuffer.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/framebuffer.fs)
set (OPENPGL_SC_GENERATED ${OPENPGL_SC_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/shaders/framebuffer_sc.h)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/samples_sc.h
    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/samples.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/samples.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/samples_sc.h
    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/samples.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/samples.fs)
set (OPENPGL_SC_GENERATED ${OPENPGL_SC_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/shaders/samples_sc.h)

#add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/simple-shader_sc.h
#    COMMAND sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/simple-shader.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/simple-shader.fs > ${CMAKE_CURRENT_BINARY_DIR}/shaders/simple-shader_sc.h
#    DEPENDS sc ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/simple-shader.vs ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/simple-shader.fs)
#set (OPENPGL_SC_GENERATED ${OPENPGL_SC_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/shaders/simple-shader_sc.h)

add_custom_target (openpgl_sc_generated DEPENDS ${OPENPGL_SC_GENERATED})

set_property (TARGET sc PROPERTY FOLDER "cmd")


######################
# openpgl-viewer

set(OPENPGL_VIEWER_SRC
    src/views/View.cpp
    src/views/CameraView.cpp
    src/views/SampleView.cpp
    src/views/SamplingDistributionView.cpp
    src/views/GuidingFieldView.cpp
    src/views/ViewportView.cpp
    src/data/Data.cpp
    src/GLShader.cpp
    src/FileManager.cpp
    src/FrameBuffer.cpp
    src/PickBuffer.cpp
    src/Application.cpp
    src/main.cpp
    third-party/ImGuiFileDialog/ImGuiFileDialog.cpp
)

add_executable(openpgl-viewer
${OPENPGL_VIEWER_SRC})

add_dependencies(openpgl-viewer openpgl_sc_generated)
add_dependencies(openpgl-viewer openpgl)

include_directories(openpgl)

target_compile_definitions(openpgl-viewer PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_include_directories(openpgl-viewer PUBLIC "./third-party/imgui" "./third-party/imgui/backends" "./third-party/ImGuiFileDialog" "./third-party/json/include")
target_include_directories(openpgl-viewer PUBLIC "./third-party/glm")
target_include_directories(openpgl-viewer PUBLIC "./third-party/colormap-shaders/include")
target_include_directories(openpgl-viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(openpgl-viewer imgui GL GLEW openpgl TBB::tbb ${CMAKE_DL_LIBS})

INSTALL(TARGETS openpgl-viewer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
