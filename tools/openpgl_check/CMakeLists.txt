add_executable(openpgl_check openpgl_check.cpp)
target_link_libraries(openpgl_check PRIVATE openpgl)
target_link_libraries(openpgl_check PRIVATE TBB::${OPENPGL_TBB_COMPONENT} )
add_dependencies(openpgl_check openpgl)

INSTALL(TARGETS openpgl_check RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


