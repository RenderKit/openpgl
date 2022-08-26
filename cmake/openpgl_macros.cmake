## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0


macro(openpgl_install_library name)
  set_target_properties(${name}
    PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  openpgl_install_target(${name})
endmacro()

macro(openpgl_install_target name)
  install(TARGETS ${name}
    EXPORT ${PROJECT_NAME}_Exports
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_SKIP
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  install(EXPORT ${PROJECT_NAME}_Exports
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
    NAMESPACE openpgl::
  )

  install(TARGETS ${name}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_ONLY
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endmacro()