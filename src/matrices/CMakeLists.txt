SET( headers tnlMatrix.h
             tnlDenseMatrix.h
             tnlTridiagonalMatrix.h
             tnlMultidiagonalMatrix.h
             tnlSparseMatrix.h
             tnlEllpackMatrix.h
             tnlSlicedEllpackMatrix.h
             tnlChunkedEllpackMatrix.h
             tnlCSRMatrix.h 
             tnlMatrixReader.h
             tnlMatrixWriter.h
             tnlMatrixSetter.h
             tnlSparseMatrixRow.h
             tnlDenseMatrixRow.h
             tnlTridiagonalMatrixRow.h
             tnlMultidiagonalMatrixSetter.h
             tnlMultidiagonalMatrixRow.h )
	
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/matrices )
set( common_SOURCES  )

SET( tnl_matrices_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
   SET( tnl_matrices_CUDA__SOURCES 
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/matrices )