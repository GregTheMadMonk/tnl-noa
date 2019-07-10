if( ${BUILD_MPI} )
   ADD_EXECUTABLE( MpiCommunicatorTest MpiCommunicatorTest.cpp )
   TARGET_COMPILE_OPTIONS( MpiCommunicatorTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MpiCommunicatorTest ${GTEST_BOTH_LIBRARIES} )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/MpiCommunicatorTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME MpiCommunicatorTest COMMAND "mpirun" ${mpi_test_parameters})

endif()
