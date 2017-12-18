find_package( PythonInterp 3 )

if( PYTHONINTERP_FOUND )
   CONFIGURE_FILE( "__init__.py.in" "${PROJECT_BUILD_PATH}/Python/__init__.py" )
   INSTALL( FILES ${PROJECT_BUILD_PATH}/Python/__init__.py
            DESTINATION lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/TNL )
endif()
