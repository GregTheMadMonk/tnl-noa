IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayAllocation ArrayAllocation.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayAllocation > ArrayAllocation.out OUTPUT ArrayAllocation.out )
   CUDA_ADD_EXECUTABLE( ArrayBinding-1 ArrayBinding-1.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayBinding-1 > ArrayBinding-1.out OUTPUT ArrayBinding-1.out )
   CUDA_ADD_EXECUTABLE( ArrayBinding-2 ArrayBinding-2.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayBinding-2 > ArrayBinding-2.out OUTPUT ArrayBinding-2.out )
   CUDA_ADD_EXECUTABLE( ArrayBinding-3 ArrayBinding-3.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayBinding-3 > ArrayBinding-3.out OUTPUT ArrayBinding-3.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsArrays-cuda ALL DEPENDS
   ArrayAllocation.out
   ArrayBinding-1.out
   ArrayBinding-2.out
   ArrayBinding-3.out )
ENDIF()

# set input and output files
set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# request to configure the file
configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

# note the option ALL which allows to build the docs together with the application
add_custom_target( doc_doxygen_tutorial_1 ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM )

INSTALL( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/tnl/html/Tutorials/Arrays )