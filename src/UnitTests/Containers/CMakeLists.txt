ADD_EXECUTABLE( ListTest ListTest.cpp )
TARGET_COMPILE_OPTIONS( ListTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ListTest ${GTEST_BOTH_LIBRARIES} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayOperationsTest ArrayOperationsTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest ${GTEST_BOTH_LIBRARIES} )
ELSE( BUILD_CUDA )
   ADD_EXECUTABLE( ArrayOperationsTest ArrayOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( ArrayOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( ArrayTest ArrayTest.cpp )
TARGET_COMPILE_OPTIONS( ArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ArrayTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( ArrayViewTest ArrayViewTest.cpp )
TARGET_COMPILE_OPTIONS( ArrayViewTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ArrayViewTest ${GTEST_BOTH_LIBRARIES} )

# NOTE: Vector = Array + VectorOperations, VectorView = ArrayView + VectorOperations,
# so we test Vector, VectorView and VectorOperations at the same time
ADD_EXECUTABLE( VectorTest-1 VectorTest-1.cpp )
ADD_EXECUTABLE( VectorTest-2 VectorTest-2.cpp )
ADD_EXECUTABLE( VectorTest-3 VectorTest-3.cpp )
TARGET_COMPILE_OPTIONS( VectorTest-1 PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_COMPILE_OPTIONS( VectorTest-2 PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_COMPILE_OPTIONS( VectorTest-3 PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( VectorTest-1 ${GTEST_BOTH_LIBRARIES} )
TARGET_LINK_LIBRARIES( VectorTest-2 ${GTEST_BOTH_LIBRARIES} )
TARGET_LINK_LIBRARIES( VectorTest-3 ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( BinaryVectorOperationsTest VectorBinaryOperationsTest.cpp )
TARGET_COMPILE_OPTIONS( BinaryVectorOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( BinaryVectorOperationsTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( UnaryVectorOperationsTest VectorUnaryOperationsTest.cpp )
TARGET_COMPILE_OPTIONS( UnaryVectorOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( UnaryVectorOperationsTest ${GTEST_BOTH_LIBRARIES} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayTestCuda ArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( ArrayViewTestCuda ArrayViewTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayViewTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( VectorTestCuda-1 VectorTest-1.cu OPTIONS ${CXX_TESTS_FLAGS} )
   CUDA_ADD_EXECUTABLE( VectorTestCuda-2 VectorTest-2.cu OPTIONS ${CXX_TESTS_FLAGS} )
   CUDA_ADD_EXECUTABLE( VectorTestCuda-3 VectorTest-3.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorTestCuda-1 ${GTEST_BOTH_LIBRARIES} )
   TARGET_LINK_LIBRARIES( VectorTestCuda-2 ${GTEST_BOTH_LIBRARIES} )
   TARGET_LINK_LIBRARIES( VectorTestCuda-3 ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( BinaryVectorOperationsTestCuda VectorBinaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BinaryVectorOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( UnaryVectorOperationsTestCuda VectorUnaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( UnaryVectorOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( MultireductionTest MultireductionTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultireductionTest ${GTEST_BOTH_LIBRARIES} )
ELSE( BUILD_CUDA )
   ADD_EXECUTABLE( MultireductionTest MultireductionTest.cpp )
   TARGET_COMPILE_OPTIONS( MultireductionTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultireductionTest ${GTEST_BOTH_LIBRARIES} )
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


ADD_TEST( ListTest ${EXECUTABLE_OUTPUT_PATH}/ListTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ArrayOperationsTest ${EXECUTABLE_OUTPUT_PATH}/ArrayOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ArrayTest ${EXECUTABLE_OUTPUT_PATH}/ArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ArrayViewTest ${EXECUTABLE_OUTPUT_PATH}/ArrayViewTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorTest-1 ${EXECUTABLE_OUTPUT_PATH}/VectorTest-1${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorTest-2 ${EXECUTABLE_OUTPUT_PATH}/VectorTest-2${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorTest-3 ${EXECUTABLE_OUTPUT_PATH}/VectorTest-3${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( BinaryVectorOperationsTest ${EXECUTABLE_OUTPUT_PATH}/BinaryVectorOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( UnaryVectorOperationsTest ${EXECUTABLE_OUTPUT_PATH}/UnaryVectorOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
IF( BUILD_CUDA )
   ADD_TEST( ArrayTestCuda ${EXECUTABLE_OUTPUT_PATH}/ArrayTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( ArrayViewTestCuda ${EXECUTABLE_OUTPUT_PATH}/ArrayViewTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorTestCuda-1 ${EXECUTABLE_OUTPUT_PATH}/VectorTestCuda-1${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorTestCuda-2 ${EXECUTABLE_OUTPUT_PATH}/VectorTestCuda-2${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( VectorTestCuda-3 ${EXECUTABLE_OUTPUT_PATH}/VectorTestCuda-3${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( BinaryVectorOperationsTestCuda ${EXECUTABLE_OUTPUT_PATH}/BinaryVectorOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
   ADD_TEST( UnaryVectorOperationsTestCuda ${EXECUTABLE_OUTPUT_PATH}/UnaryVectorOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX} )
ENDIF()
ADD_TEST( MultireductionTest ${EXECUTABLE_OUTPUT_PATH}/MultireductionTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticArrayTest ${EXECUTABLE_OUTPUT_PATH}/StaticArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticVectorTest ${EXECUTABLE_OUTPUT_PATH}/StaticVectorTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticVectorOperationsTest ${EXECUTABLE_OUTPUT_PATH}/StaticVectorOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )


ADD_SUBDIRECTORY( Multimaps )


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

   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedVectorBinaryOperationsTestCuda DistributedVectorBinaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedVectorUnaryOperationsTestCuda DistributedVectorUnaryOperationsTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )
   endif( BUILD_CUDA )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedArrayTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorBinaryOperationsTest COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorUnaryOperationsTest COMMAND "mpirun" ${mpi_test_parameters})

   if( BUILD_CUDA )
      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorTestCuda COMMAND "mpirun" ${mpi_test_parameters})

      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorBinaryOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})

      SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorUnaryOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})
   endif()
endif()
