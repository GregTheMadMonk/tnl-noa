if( ${CXX_COMPILER_NAME} STREQUAL "mpic++" )
ADD_EXECUTABLE( DistributedGridTest_1D DistributedGridTest_1D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_1D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_1D
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

ADD_EXECUTABLE( DistributedGridTest_2D DistributedGridTest_2D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_2D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_2D
                              ${GTEST_BOTH_LIBRARIES}
                              tnl)

ADD_EXECUTABLE( DistributedGridTest_3D DistributedGridTest_3D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_3D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_3D
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

ADD_EXECUTABLE( CopyEntitesTest CopyEntitiesTest.cpp )
   TARGET_COMPILE_OPTIONS( CopyEntitesTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CopyEntitesTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

ADD_EXECUTABLE( DistributedGridIOTest DistributedGridIOTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridIOTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridIOTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )

ADD_TEST( NAME CopyEntitesTest COMMAND ${EXECUTABLE_OUTPUT_PATH}/CopyEntitesTest${CMAKE_EXECUTABLE_SUFFIX} )
   
SET (mpi_test_parameters_1d -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridTest_1D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_1D COMMAND "mpirun" ${mpi_test_parameters_1d})

SET (mpi_test_parameters_2d -np 9 -H localhost:9 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridTest_2D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_2D COMMAND "mpirun" ${mpi_test_parameters_2d})

SET (mpi_test_parameters_3d -np 27 -H localhost:27 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridTest_3D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_3D COMMAND "mpirun" ${mpi_test_parameters_3d})

SET (mpi_test_parameters_IO -np 4 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridIOTest${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridIOTest COMMAND "mpirun" ${mpi_test_parameters_IO})

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( GPUDistributedGridIOTest GPUDistributedGridIOTest.cu OPTIONS ${CXX_TESTS_FLAGS})
    TARGET_LINK_LIBRARIES( GPUDistributedGridIOTest 
                                    ${GTEST_BOTH_LIBRARIES}
                                    tnl )   
ENDIF( BUILD_CUDA )  

endif()

