find_package(ZLIB)


add_executable(openpgl_gpu_cpu openpgl_gpu.cpp)
target_link_libraries(openpgl_gpu_cpu PRIVATE openpgl)
target_link_libraries(openpgl_gpu_cpu PRIVATE TBB::${OPENPGL_TBB_COMPONENT} ZLIB::ZLIB)
target_compile_definitions(openpgl_gpu_cpu PRIVATE OPENPGL_GPU_CPU)
#target_link_options(openpgl_gpu_cpu PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
#target_compile_options(openpgl_gpu_cpu PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
add_dependencies(openpgl_gpu_cpu openpgl)

INSTALL(TARGETS openpgl_gpu_cpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


add_executable(openpgl_gpu_sycl openpgl_gpu.cpp)
target_link_libraries(openpgl_gpu_sycl PRIVATE openpgl)
target_link_libraries(openpgl_gpu_sycl PRIVATE TBB::${OPENPGL_TBB_COMPONENT} ZLIB::ZLIB)
target_compile_definitions(openpgl_gpu_sycl PRIVATE OPENPGL_GPU_SYCL)
target_link_options(openpgl_gpu_sycl PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
target_compile_options(openpgl_gpu_sycl PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
add_dependencies(openpgl_gpu_sycl openpgl)

INSTALL(TARGETS openpgl_gpu_sycl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
