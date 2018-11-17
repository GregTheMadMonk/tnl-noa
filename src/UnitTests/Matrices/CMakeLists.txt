IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.h SparseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest SparseMatrixTest.h SparseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.h DenseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.h SparseMatrixCopyTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest SparseMatrixTest.h SparseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.h DenseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( DenseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )


ADD_TEST( SparseMatrixCopyTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixCopyTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( DenseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )


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
