IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( Expressions Expressions.cu )
   ADD_CUSTOM_COMMAND( COMMAND Expressions > Expressions.out OUTPUT Expressions.out )
   #ADD_EXECUTABLE( Reduction Reduction.cpp )
   CUDA_ADD_EXECUTABLE( Reduction Reduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND Reduction > Reduction.out OUTPUT Reduction.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsVectors-cuda ALL DEPENDS
   Expressions.out
   Reduction.out )
ENDIF()

# set input and output files
set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# request to configure the file
configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

# note the option ALL which allows to build the docs together with the application
add_custom_target( doc_doxygen_tutorial_vectors ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM )

INSTALL( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/tnl/html/Tutorials/Vectors )