ADD_EXECUTABLE( DistributedGridTest_1D${mpiExt}${debugExt} DistributedGridTest_1D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_1D${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_1D${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( DistributedGridTest_2D${mpiExt}${debugExt} DistributedGridTest_2D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_2D${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_2D${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( DistributedGridTest_3D${mpiExt}${debugExt} DistributedGridTest_3D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_3D${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_3D${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( CopyEntitesTest${mpiExt}${debugExt} CopyEntitiesTest.cpp )
   TARGET_COMPILE_OPTIONS( CopyEntitesTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CopyEntitesTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( DistributedGridIOTest${mpiExt}${debugExt} DistributedGridIOTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridIOTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridIOTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
    
SET (mpi_test_parameters_1d -np 4 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridTest_1D${mpiExt}${debugExt}")
ADD_TEST( NAME DistributedGridTest_1D COMMAND "mpirun" ${mpi_test_parameters_1d})

SET (mpi_test_parameters_2d -np 9 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridTest_2D${mpiExt}${debugExt}")
ADD_TEST( NAME DistributedGridTest_2D COMMAND "mpirun" ${mpi_test_parameters_2d})

SET (mpi_test_parameters_3d -np 27 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridTest_3D${mpiExt}${debugExt}")
ADD_TEST( NAME DistributedGridTest_3D COMMAND "mpirun" ${mpi_test_parameters_3d})

ADD_TEST( CopyEntitesTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/CopyEntitesTest${mpiExt}${debugExt} )

SET (mpi_test_parameters_IO -np 4 "${EXECUTABLE_OUTPUT_PATH}/DistributedGridIOTest${mpiExt}${debugExt}")
ADD_TEST( NAME DistributedGridIOTest COMMAND "mpirun" ${mpi_test_parameters_IO})
