SET( headers tnlMatrix.h
             tnlMatrix_impl.h
             tnlDenseMatrix.h
             tnlDenseMatrix_impl.h
             tnlTridiagonalMatrix.h
             tnlTridiagonalMatrix_impl.h
             tnlMultidiagonalMatrix.h
             tnlMultidiagonalMatrix_impl.h
             tnlSparseMatrix.h
             tnlSparseMatrix_impl.h
             tnlEllpackMatrix.h
             tnlEllpackMatrix_impl.h
             tnlSlicedEllpackMatrix.h
             tnlSlicedEllpackMatrix_impl.h
             tnlChunkedEllpackMatrix.h
             tnlChunkedEllpackMatrix_impl.h
             tnlCSRMatrix.h
             tnlCSRMatrix_impl.h 
             tnlMatrixReader.h
             tnlMatrixReader_impl.h
             tnlMatrixWriter.h
             tnlMatrixWriter_impl.h
             tnlMatrixSetter.h
             tnlMatrixSetter_impl.h
             tnlSparseMatrixRow.h
             tnlSparseMatrixRow_impl.h
             tnlDenseMatrixRow.h
             tnlDenseMatrixRow_impl.h
             tnlTridiagonalMatrixRow.h
             tnlTridiagonalMatrixRow_impl.h
             tnlMultidiagonalMatrixSetter.h
             tnlMultidiagonalMatrixSetter_impl.h
             tnlMultidiagonalMatrixRow.h
             tnlMultidiagonalMatrixRow_impl.h  )

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
source_group("matrices" FILES ${headers})