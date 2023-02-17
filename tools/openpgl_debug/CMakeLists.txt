add_executable(openpgl_debug openpgl_debug.cpp)
target_link_libraries(openpgl_debug PRIVATE openpgl)
target_link_libraries(openpgl_debug PRIVATE TBB::${OPENPGL_TBB_COMPONENT} )
add_dependencies(openpgl_debug openpgl)

INSTALL(TARGETS openpgl_debug RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


