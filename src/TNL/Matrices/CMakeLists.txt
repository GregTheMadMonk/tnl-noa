SET( headers Matrix.h
             Matrix_impl.h
             DenseMatrix.h
             DenseMatrix_impl.h
             TridiagonalMatrix.h
             TridiagonalMatrix_impl.h
             MultidiagonalMatrix.h
             MultidiagonalMatrix_impl.h
             SparseMatrix.h
             SparseMatrix_impl.h
             EllpackMatrix.h
             EllpackMatrix_impl.h
             SlicedEllpackMatrix.h
             SlicedEllpackMatrix_impl.h
             ChunkedEllpackMatrix.h
             ChunkedEllpackMatrix_impl.h
             CSRMatrix.h
             CSRMatrix_impl.h 
             MatrixReader.h
             MatrixReader_impl.h
             MatrixWriter.h
             MatrixWriter_impl.h
             MatrixSetter.h
             MatrixSetter_impl.h
             SparseMatrixRow.h
             SparseMatrixRow_impl.h
             DenseMatrixRow.h
             DenseMatrixRow_impl.h
             TridiagonalMatrixRow.h
             TridiagonalMatrixRow_impl.h
             MultidiagonalMatrixSetter.h
             MultidiagonalMatrixSetter_impl.h
             MultidiagonalMatrixRow.h
             MultidiagonalMatrixRow_impl.h  )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/matrices )
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