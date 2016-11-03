SET( headers tnlMatrix_impl.h
             tnlDenseMatrix_impl.h
             tnlTridiagonalMatrix_impl.h
             tnlMultidiagonalMatrix_impl.h 
             tnlSparseMatrix_impl.h
             tnlEllpackMatrix_impl.h
             tnlEllpackSymMatrix_impl.h
             tnlSlicedEllpackMatrix_impl.h
             tnlChunkedEllpackMatrix_impl.h
             tnlCSRMatrix_impl.h
             tnlMatrixReader_impl.h
             tnlMatrixWriter_impl.h
             tnlMatrixSetter_impl.h
             tnlBiEllpackMatrix_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/matrices )
set( common_SOURCES
     #${CURRENT_DIR}/tnlDenseMatrix_impl.cpp
 )       

IF( BUILD_CUDA )
   set( tnl_implementation_matrices_CUDA__SOURCES
        ${common_SOURCES} 
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_matrices_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/matrices )

