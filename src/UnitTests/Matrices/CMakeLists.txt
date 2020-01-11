ADD_SUBDIRECTORY( Legacy )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest SparseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixTest TridiagonalMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( TridiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

#   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixTest MultidiagonalMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MultidiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_CSR_segments SparseMatrixTest_CSR_segments.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR_segments ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_Ellpack_segments SparseMatrixTest_Ellpack_segments.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack_segments ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_SlicedEllpack_segments SparseMatrixTest_SlicedEllpack_segments.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_SlicedEllpack_segments ${GTEST_BOTH_LIBRARIES} )

ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest SparseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( DenseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( TridiagonalMatrixTest TridiagonalMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( TridiagonalMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( TridiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

#   ADD_EXECUTABLE( MultidiagonalMatrixTest MultidiagonalMatrixTest.cpp )
#   TARGET_COMPILE_OPTIONS( MultidiagonalMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MultidiagonalMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_CSR_segments SparseMatrixTest_CSR_segments.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_CSR_segments PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR_segments ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_Ellpack_segments SparseMatrixTest_Ellpack_segments.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_Ellpack_segments PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack_segments ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_SlicedEllpack_segments SparseMatrixTest_SlicedEllpack_segments.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_SlicedEllpack_segments PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_SlicedEllpack_segments ${GTEST_BOTH_LIBRARIES} )

ENDIF( BUILD_CUDA )


ADD_TEST( SparseMatrixCopyTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixCopyTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( DenseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/DenseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( TridiagonalMatrixTest ${EXECUTABLE_OUTPUT_PATH}/TridiagonalMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( MultidiagonalMatrixTest ${EXECUTABLE_OUTPUT_PATH}/MultidiagonalMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )

ADD_TEST( SparseMatrixTest_CSR_segments ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_CSR_segments${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_Ellpack_segments ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_Ellpack_segments${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_SlicedEllpack_segments ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_SlicedEllpack_segments${CMAKE_EXECUTABLE_SUFFIX} )

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
