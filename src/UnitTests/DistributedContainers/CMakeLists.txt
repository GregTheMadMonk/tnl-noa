if( ${BUILD_MPI} )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedArrayTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
else()
   ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedArrayTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
endif()

SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )
ADD_TEST( NAME DistributedArrayTest COMMAND "mpirun" ${mpi_test_parameters})

endif()
