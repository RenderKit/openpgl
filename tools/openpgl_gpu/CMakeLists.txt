find_package(ZLIB)

add_executable(openpgl_gpu openpgl_gpu.cpp)
target_link_libraries(openpgl_gpu PRIVATE openpgl)
target_link_libraries(openpgl_gpu PRIVATE TBB::${OPENPGL_TBB_COMPONENT} ZLIB::ZLIB)
add_dependencies(openpgl_gpu openpgl)

INSTALL(TARGETS openpgl_gpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


