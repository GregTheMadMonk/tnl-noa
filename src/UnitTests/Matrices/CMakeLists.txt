IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest SparseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_AdEllpack SparseMatrixTest_AdEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_AdEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_BiEllpack SparseMatrixTest_BiEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_ChunkedEllpack SparseMatrixTest_ChunkedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_CSR SparseMatrixTest_CSR.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_Ellpack SparseMatrixTest_Ellpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_SlicedEllpack SparseMatrixTest_SlicedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ####
   # Segments tests
   CUDA_ADD_EXECUTABLE( SparseMatrixTest_CSR_segments SparseMatrixTest_CSR_segments.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR_segments ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SparseMatrixTest_Ellpack_segments SparseMatrixTest_Ellpack_segments.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack_segments ${GTEST_BOTH_LIBRARIES} )

ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( SparseMatrixCopyTest SparseMatrixCopyTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest SparseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_AdEllpack SparseMatrixTest_AdEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_AdEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_AdEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_BiEllpack SparseMatrixTest_BiEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_BiEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_ChunkedEllpack SparseMatrixTest_ChunkedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_ChunkedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_CSR SparseMatrixTest_CSR.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_CSR PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_Ellpack SparseMatrixTest_Ellpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_Ellpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_SlicedEllpack SparseMatrixTest_SlicedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_SlicedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( DenseMatrixTest DenseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( DenseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DenseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ####
   # Segments tests
   ADD_EXECUTABLE( SparseMatrixTest_CSR_segments SparseMatrixTest_CSR_segments.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_CSR_segments PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_CSR_segments ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SparseMatrixTest_Ellpack_segments SparseMatrixTest_Ellpack_segments.cpp )
   TARGET_COMPILE_OPTIONS( SparseMatrixTest_Ellpack_segments PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SparseMatrixTest_Ellpack_segments ${GTEST_BOTH_LIBRARIES} )


ENDIF( BUILD_CUDA )


ADD_TEST( SparseMatrixCopyTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixCopyTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
# TODO: Uncomment the following when AdEllpack works
#ADD_TEST( SparseMatrixTest_AdEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_AdEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_BiEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_BiEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_ChunkedEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_ChunkedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_CSR ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_CSR${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_Ellpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_Ellpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_SlicedEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_SlicedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
# TODO: DenseMatrixTest is not finished
#ADD_TEST( DenseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/DenseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )

####
# Segments tests
ADD_TEST( SparseMatrixTest_CSR_segments ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_CSR_segments${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SparseMatrixTest_Ellpack_segments ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_Ellpack_segments${CMAKE_EXECUTABLE_SUFFIX} )

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
