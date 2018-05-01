find_package( PythonInterp 3 )
find_package( PythonLibs 3 )

set( PYTHON_SITE_PACKAGES_DIR lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages )

if( PYTHONINTERP_FOUND )
   CONFIGURE_FILE( "__init__.py.in" "${PROJECT_BUILD_PATH}/Python/__init__.py" )
   INSTALL( FILES ${PROJECT_BUILD_PATH}/Python/__init__.py
                  LogParser.py
            DESTINATION ${PYTHON_SITE_PACKAGES_DIR}/TNL )
endif()

if( PYTHONLIBS_FOUND )
   include(ExternalProject)
   ExternalProject_Add(pybind11_project
     GIT_REPOSITORY    https://github.com/pybind/pybind11.git
     GIT_TAG           master
     SOURCE_DIR        "${CMAKE_BINARY_DIR}/pybind11-src"
     BINARY_DIR        "${CMAKE_BINARY_DIR}/pybind11-build"
     CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DPYBIND11_TEST=FALSE
   )
   add_subdirectory(${CMAKE_BINARY_DIR}/pybind11-src ${CMAKE_BINARY_DIR}/pybind11-build)
#   find_package(pybind11 REQUIRED
#                PATHS ${CMAKE_INSTALL_PREFIX})

   add_subdirectory(pytnl)
else()
   message( "The Python.h header file was not found, Python bindings will not be builg." )
endif()
