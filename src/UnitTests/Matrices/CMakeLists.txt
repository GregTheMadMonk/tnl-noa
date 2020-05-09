ADD_SUBDIRECTORY( Legacy )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( DenseMatrixCopyTest DenseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixTest TridiagonalMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( TridiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixTest MultidiagonalMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultidiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_CSR SparseMatrixTest_CSR.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_Ellpack SparseMatrixTest_Ellpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_SlicedEllpack SparseMatrixTest_SlicedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_ChunkedEllpack SparseMatrixTest_ChunkedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_BiEllpack SparseMatrixTest_BiEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( BinarySparseMatrixTest_CSR BinarySparseMatrixTest_CSR.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( BinarySparseMatrixTest_Ellpack BinarySparseMatrixTest_Ellpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( BinarySparseMatrixTest_SlicedEllpack BinarySparseMatrixTest_SlicedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( BinarySparseMatrixCopyTest BinarySparseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SymmetricSparseMatrixTest_CSR SymmetricSparseMatrixTest_CSR.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SymmetricSparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( LambdaMatrixTest LambdaMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( LambdaMatrixTest ${GTEST_BOTH_LIBRARIES} )

ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( DenseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( DenseMatrixCopyTest DenseMatrixCopyTest.cpp )
   TARGET_COMPILE_OPTIONS( DenseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( TridiagonalMatrixTest TridiagonalMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( TridiagonalMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( TridiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( MultidiagonalMatrixTest MultidiagonalMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( MultidiagonalMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultidiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_CSR SparseMatrixTest_CSR.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_CSR PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_Ellpack SparseMatrixTest_Ellpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_Ellpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_SlicedEllpack SparseMatrixTest_SlicedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_SlicedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_ChunkedEllpack SparseMatrixTest_ChunkedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_ChunkedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_BiEllpack SparseMatrixTest_BiEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_BiEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( BinarySparseMatrixTest_CSR BinarySparseMatrixTest_CSR.cpp )
   TARGET_COMPILE_OPTIONS( BinarySparseMatrixTest_CSR PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( BinarySparseMatrixTest_Ellpack BinarySparseMatrixTest_Ellpack.cpp )
   TARGET_COMPILE_OPTIONS( BinarySparseMatrixTest_Ellpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( BinarySparseMatrixTest_SlicedEllpack BinarySparseMatrixTest_SlicedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( BinarySparseMatrixTest_SlicedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( BinarySparseMatrixCopyTest BinarySparseMatrixCopyTest.cpp )
   TARGET_COMPILE_OPTIONS( BinarySparseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinarySparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SymmetricSparseMatrixTest_CSR SymmetricSparseMatrixTest_CSR.cpp )
   TARGET_COMPILE_OPTIONS( SymmetricSparseMatrixTest_CSR PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SymmetricSparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( LambdaMatrixTest LambdaMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( LambdaMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( LambdaMatrixTest ${GTEST_BOTH_LIBRARIES} )

ENDIF( BUILD_CUDA )

ADD_TEST( DenseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/DenseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( TridiagonalMatrixTest ${EXECUTABLE_OUTPUT_PATH}/TridiagonalMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MultidiagonalMatrixTest ${EXECUTABLE_OUTPUT_PATH}/MultidiagonalMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )

ADD_TEST( SparseMatrixTest_CSR ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_CSR${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_Ellpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_Ellpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_SlicedEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_SlicedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_ChunkedEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_ChunkedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_BiEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_BiEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixCopyTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixCopyTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( BinarySparseMatrixTest_CSR ${EXECUTABLE_OUTPUT_PATH}/BinarySparseMatrixTest_CSR${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( BinarySparseMatrixTest_Ellpack ${EXECUTABLE_OUTPUT_PATH}/BinarySparseMatrixTest_Ellpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( BinarySparseMatrixTest_SlicedEllpack ${EXECUTABLE_OUTPUT_PATH}/BinarySparseMatrixTest_SlicedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( BinarySparseMatrixCopyTest ${EXECUTABLE_OUTPUT_PATH}/BinarySparseMatrixCopyTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SymmetricSparseMatrixTest_CSR ${EXECUTABLE_OUTPUT_PATH}/SymmetricSparseMatrixTest_CSR${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( LambdaMatrixTest ${EXECUTABLE_OUTPUT_PATH}/LambdaMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )

if( ${BUILD_MPI} )
   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedMatrixTest DistributedMatrixTest.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedMatrixTest ${GTEST_BOTH_LIBRARIES} )
   else()
      ADD_EXECUTABLE( DistributedMatrixTest DistributedMatrixTest.cpp )
      TARGET_COMPILE_OPTIONS( DistributedMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedMatrixTest ${GTEST_BOTH_LIBRARIES} )
   endif()

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedMatrixTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedMatrixTest COMMAND "mpirun" ${mpi_test_parameters})
endif()
