add_executable(openpgl_bench openpgl_bench.cpp)
target_link_libraries(openpgl_bench PRIVATE openpgl)
target_link_libraries(openpgl_bench PRIVATE TBB::${OPENPGL_TBB_COMPONENT} )
add_dependencies(openpgl_bench openpgl)

INSTALL(TARGETS openpgl_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


