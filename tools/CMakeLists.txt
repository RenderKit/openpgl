if(OPENPGL_BUILD_CHECK_TOOL)
    add_subdirectory(openpgl_check)
endif()

if(OPENPGL_BUILD_TOOLS)
    add_subdirectory(openpgl_debug)
    add_subdirectory(openpgl_bench)
endif()