if( ${BUILD_MPI} )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedArrayTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

   CUDA_ADD_EXECUTABLE( DistributedVectorTest DistributedVectorTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

   CUDA_ADD_EXECUTABLE( DistributedMatrixTest DistributedMatrixTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedMatrixTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
else()
   ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedArrayTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

   ADD_EXECUTABLE( DistributedVectorTest DistributedVectorTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

   ADD_EXECUTABLE( DistributedMatrixTest DistributedMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedMatrixTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
endif()

SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )
ADD_TEST( NAME DistributedArrayTest COMMAND "mpirun" ${mpi_test_parameters})

SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedVectorTest${CMAKE_EXECUTABLE_SUFFIX}" )
ADD_TEST( NAME DistributedVectorTest COMMAND "mpirun" ${mpi_test_parameters})

SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedMatrixTest${CMAKE_EXECUTABLE_SUFFIX}" )
ADD_TEST( NAME DistributedMatrixTest COMMAND "mpirun" ${mpi_test_parameters})

endif()
