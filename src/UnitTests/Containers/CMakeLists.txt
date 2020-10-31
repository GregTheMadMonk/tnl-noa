ADD_SUBDIRECTORY( ndarray )
ADD_SUBDIRECTORY( Segments )

set( CPP_TESTS
         ArrayTest
         ArrayViewTest
         StaticArrayTest
         StaticVectorTest
         StaticVectorOperationsTest
         StaticVectorOfStaticVectorsTest
         VectorTest
         VectorPrefixSumTest
         VectorEvaluateAndReduceTest
         VectorBinaryOperationsTest
         VectorUnaryOperationsTest
         VectorVerticalOperationsTest
         VectorOfStaticVectorsTest
)
set( CUDA_TESTS
         ArrayTestCuda
         ArrayViewTestCuda
         VectorTestCuda
         VectorPrefixSumTestCuda
         VectorEvaluateAndReduceTestCuda
         VectorBinaryOperationsTestCuda
         VectorUnaryOperationsTestCuda
         VectorVerticalOperationsTestCuda
         # FIXME: fails due to unspecified launch failure in the CUDA reduction kernel for scalar product,
         #        see https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/-/issues/82
         #VectorOfStaticVectorsTestCuda
)

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${GTEST_BOTH_LIBRARIES} )
   add_test( ${target} ${EXECUTABLE_OUTPUT_PATH}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      cuda_add_executable( ${target} ${target}.cu OPTIONS ${CXX_TESTS_FLAGS} )
      target_link_libraries( ${target} ${GTEST_BOTH_LIBRARIES} )
      add_test( ${target} ${EXECUTABLE_OUTPUT_PATH}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()


if( ${BUILD_MPI} )
   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTestCuda.cu
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
      CUDA_ADD_EXECUTABLE( DistributedVectorTestCuda DistributedVectorTestCuda.cu
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
      CUDA_ADD_EXECUTABLE( DistributedVectorBinaryOperationsTestCuda DistributedVectorBinaryOperationsTestCuda.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedVectorUnaryOperationsTestCuda DistributedVectorUnaryOperationsTestCuda.cu OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTestCuda ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedVectorVerticalOperationsTestCuda DistributedVectorVerticalOperationsTestCuda.cu OPTIONS ${CXX_TESTS_FLAGS} )
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
