ADD_EXECUTABLE( ArrayTest ArrayTest.cpp )
TARGET_COMPILE_OPTIONS( ArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ArrayTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( ArrayViewTest ArrayViewTest.cpp )
TARGET_COMPILE_OPTIONS( ArrayViewTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ArrayViewTest ${GTEST_BOTH_LIBRARIES} )

# NOTE: Vector = Array + VectorOperations, VectorView = ArrayView + VectorOperations,
# so we test Vector, VectorView and VectorOperations at the same time
ADD_EXECUTABLE( VectorTest VectorTest.cpp )
ADD_EXECUTABLE( VectorPrefixSumTest VectorPrefixSumTest.cpp )
ADD_EXECUTABLE( VectorEvaluateAndReduceTest VectorEvaluateAndReduceTest.cpp )
TARGET_COMPILE_OPTIONS( VectorTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_COMPILE_OPTIONS( VectorPrefixSumTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_COMPILE_OPTIONS( VectorEvaluateAndReduceTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( VectorTest ${GTEST_BOTH_LIBRARIES} )
TARGET_LINK_LIBRARIES( VectorPrefixSumTest ${GTEST_BOTH_LIBRARIES} )
TARGET_LINK_LIBRARIES( VectorEvaluateAndReduceTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( VectorBinaryOperationsTest VectorBinaryOperationsTest.cpp )
TARGET_COMPILE_OPTIONS( VectorBinaryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( VectorBinaryOperationsTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( VectorUnaryOperationsTest VectorUnaryOperationsTest.cpp )
TARGET_COMPILE_OPTIONS( VectorUnaryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( VectorUnaryOperationsTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( VectorVerticalOperationsTest VectorVerticalOperationsTest.cpp )
TARGET_COMPILE_OPTIONS( VectorVerticalOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( VectorVerticalOperationsTest ${GTEST_BOTH_LIBRARIES} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayTestCuda ArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( ArrayViewTestCuda ArrayViewTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayViewTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( VectorTestCuda VectorTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   CUDA_ADD_EXECUTABLE( VectorPrefixSumTestCuda VectorPrefixSumTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   CUDA_ADD_EXECUTABLE( VectorEvaluateAndReduceTestCuda VectorEvaluateAndReduceTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorTestCuda ${GTEST_BOTH_LIBRARIES} )
   TARGET_LINK_LIBRARIES( VectorPrefixSumTestCuda ${GTEST_BOTH_LIBRARIES} )
   TARGET_LINK_LIBRARIES( VectorEvaluateAndReduceTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( VectorBinaryOperationsTestCuda VectorBinaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorBinaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( VectorUnaryOperationsTestCuda VectorUnaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorUnaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( VectorVerticalOperationsTestCuda VectorVerticalOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorVerticalOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticArrayTest StaticArrayTest.cpp )
TARGET_COMPILE_OPTIONS( StaticArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticArrayTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( StaticVectorTest StaticVectorTest.cpp )
TARGET_COMPILE_OPTIONS( StaticVectorTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticVectorTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( StaticVectorOperationsTest StaticVectorOperationsTest.cpp )
TARGET_COMPILE_OPTIONS( StaticVectorOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticVectorOperationsTest ${GTEST_BOTH_LIBRARIES} )


ADD_TEST( ArrayTest ${EXECUTABLE_OUTPUT_PATH}/ArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ArrayViewTest ${EXECUTABLE_OUTPUT_PATH}/ArrayViewTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorTest ${EXECUTABLE_OUTPUT_PATH}/VectorTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorPrefixSumTest ${EXECUTABLE_OUTPUT_PATH}/VectorPrefixSumTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorEvaluateAndReduceTest ${EXECUTABLE_OUTPUT_PATH}/VectorEvaluateAndReduceTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorBinaryOperationsTest ${EXECUTABLE_OUTPUT_PATH}/VectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorUnaryOperationsTest ${EXECUTABLE_OUTPUT_PATH}/VectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorVerticalOperationsTest ${EXECUTABLE_OUTPUT_PATH}/VectorVerticalOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
IF( BUILD_CUDA )
   ADD_TEST( ArrayTestCuda ${EXECUTABLE_OUTPUT_PATH}/ArrayTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( ArrayViewTestCuda ${EXECUTABLE_OUTPUT_PATH}/ArrayViewTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorTestCuda ${EXECUTABLE_OUTPUT_PATH}/VectorTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorPrefixSumTestCuda ${EXECUTABLE_OUTPUT_PATH}/VectorPrefixSumTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorEvaluateAndReduceTestCuda ${EXECUTABLE_OUTPUT_PATH}/VectorEvaluateAndReduceTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorBinaryOperationsTestCuda ${EXECUTABLE_OUTPUT_PATH}/VectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorUnaryOperationsTestCuda ${EXECUTABLE_OUTPUT_PATH}/VectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorVerticalOperationsTestCuda ${EXECUTABLE_OUTPUT_PATH}/VectorVerticalOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
ENDIF()
ADD_TEST( StaticArrayTest ${EXECUTABLE_OUTPUT_PATH}/StaticArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticVectorTest ${EXECUTABLE_OUTPUT_PATH}/StaticVectorTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticVectorOperationsTest ${EXECUTABLE_OUTPUT_PATH}/StaticVectorOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )


ADD_SUBDIRECTORY( Multimaps )
ADD_SUBDIRECTORY( ndarray )


if( ${BUILD_MPI} )
   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedArrayTest ${GTEST_BOTH_LIBRARIES} )
   else()
      ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cpp )
      TARGET_COMPILE_OPTIONS( DistributedArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedArrayTest ${GTEST_BOTH_LIBRARIES} )
   endif()

   ADD_EXECUTABLE( DistributedVectorTest DistributedVectorTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorTest ${GTEST_BOTH_LIBRARIES} )

   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedVectorTestCuda DistributedVectorTest.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorTestCuda ${GTEST_BOTH_LIBRARIES} )
   endif()

   ADD_EXECUTABLE( DistributedVectorBinaryOperationsTest DistributedVectorBinaryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorBinaryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( DistributedVectorUnaryOperationsTest DistributedVectorUnaryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorUnaryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( DistributedVectorVerticalOperationsTest DistributedVectorVerticalOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorVerticalOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorVerticalOperationsTest ${GTEST_BOTH_LIBRARIES} )

   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedVectorBinaryOperationsTestCuda DistributedVectorBinaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedVectorUnaryOperationsTestCuda DistributedVectorUnaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedVectorVerticalOperationsTestCuda DistributedVectorVerticalOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorVerticalOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )
   endif( BUILD_CUDA )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedArrayTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorBinaryOperationsTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorUnaryOperationsTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorVerticalOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorVerticalOperationsTest COMMAND "mpirun" ${mpi_test_parameters})

   if( BUILD_CUDA )
      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorTestCuda COMMAND "mpirun" ${mpi_test_parameters})

      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorBinaryOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})

      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorUnaryOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})

      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorVerticalOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorVerticalOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})
   endif()
endif()
