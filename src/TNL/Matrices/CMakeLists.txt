SET( headers Matrix.h
             Matrix_impl.h
             Dense.h
             Dense_impl.h
             Tridiagonal.h
             Tridiagonal_impl.h
             Multidiagonal.h
             Multidiagonal_impl.h
             Sparse.h
             Sparse_impl.h
             Ellpack.h
             Ellpack_impl.h
             SlicedEllpack.h
             SlicedEllpack_impl.h
             ChunkedEllpack.h
             ChunkedEllpack_impl.h
             CSR.h
             CSR_impl.h 
             MatrixReader.h
             MatrixReader_impl.h
             MatrixWriter.h
             MatrixWriter_impl.h
             MatrixSetter.h
             MatrixSetter_impl.h
             SparseRow.h
             SparseRow_impl.h
             DenseRow.h
             DenseRow_impl.h
             TridiagonalRow.h
             TridiagonalRow_impl.h
             MultidiagonalMatrixSetter.h
             MultidiagonalMatrixSetter_impl.h
             MultidiagonalRow.h
             MultidiagonalRow_impl.h  )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Matrices )
set( common_SOURCES  )

SET( tnl_matrices_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
   SET( tnl_matrices_CUDA__SOURCES 
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Matrices )