ADD_EXECUTABLE( DirectionsTest DirectionsTest.cpp )
   TARGET_COMPILE_OPTIONS( DirectionsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DirectionsTest ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( CopyEntitesTest CopyEntitiesTest.cpp )
   TARGET_COMPILE_OPTIONS( CopyEntitesTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CopyEntitesTest ${GTEST_BOTH_LIBRARIES} )

# FIXME: DistributedGrid refucktoring
#ADD_EXECUTABLE( CutMeshFunctionTest CutMeshFunctionTest.cpp )
#   TARGET_COMPILE_OPTIONS( CutMeshFunctionTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( CutMeshFunctionTest ${GTEST_BOTH_LIBRARIES} )

ADD_TEST( NAME DirectionsTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DirectionsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( NAME CopyEntitesTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CopyEntitesTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( NAME CutMeshFunctionTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX} )

if( BUILD_MPI )
ADD_EXECUTABLE( DistributedGridTest_1D DistributedGridTest_1D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_1D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_1D ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( DistributedGridTest_2D DistributedGridTest_2D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_2D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_2D ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( DistributedGridTest_3D DistributedGridTest_3D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_3D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_3D ${GTEST_BOTH_LIBRARIES} )

ADD_EXECUTABLE( CutDistributedGridTest CutDistributedGridTest.cpp )
   TARGET_COMPILE_OPTIONS( CutDistributedGridTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CutDistributedGridTest ${GTEST_BOTH_LIBRARIES} )

#ADD_EXECUTABLE( CutDistributedMeshFunctionTest CutDistributedMeshFunctionTest.cpp )
#   TARGET_COMPILE_OPTIONS( CutDistributedMeshFunctionTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( CutDistributedMeshFunctionTest ${GTEST_BOTH_LIBRARIES} )

#ADD_EXECUTABLE( DistributedVectorFieldIO_MPIIOTest DistributedVectorFieldIO_MPIIOTest.cpp )
#   TARGET_COMPILE_OPTIONS( DistributedVectorFieldIO_MPIIOTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( DistributedVectorFieldIO_MPIIOTest ${GTEST_BOTH_LIBRARIES} )

IF( BUILD_CUDA )
#    CUDA_ADD_EXECUTABLE( DistributedGridIOTest DistributedGridIOTest.cu OPTIONS ${CXX_TESTS_FLAGS})
#       TARGET_LINK_LIBRARIES( DistributedGridIOTest ${GTEST_BOTH_LIBRARIES} )

#    CUDA_ADD_EXECUTABLE( DistributedGridIO_MPIIOTest DistributedGridIO_MPIIOTest.cu OPTIONS ${CXX_TESTS_FLAGS})
#    TARGET_LINK_LIBRARIES( DistributedGridIO_MPIIOTest ${GTEST_BOTH_LIBRARIES} )
ELSE(  BUILD_CUDA )

#    ADD_EXECUTABLE( DistributedGridIO_MPIIOTest DistributedGridIO_MPIIOTest.cpp )
#       TARGET_COMPILE_OPTIONS( DistributedGridIO_MPIIOTest PRIVATE ${CXX_TESTS_FLAGS} )
#       TARGET_LINK_LIBRARIES( DistributedGridIO_MPIIOTest ${GTEST_BOTH_LIBRARIES} )

#    ADD_EXECUTABLE( DistributedGridIOTest DistributedGridIOTest.cpp )
#       TARGET_COMPILE_OPTIONS( DistributedGridIOTest PRIVATE ${CXX_TESTS_FLAGS} )
#       TARGET_LINK_LIBRARIES( DistributedGridIOTest ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )

if( BUILD_CUDA )
   cuda_add_executable( DistributedMeshTest DistributedMeshTest.cu OPTIONS ${CXX_TESTS_FLAGS})
   target_link_libraries( DistributedMeshTest ${GTEST_BOTH_LIBRARIES} )
else()
   add_executable( DistributedMeshTest DistributedMeshTest.cpp )
   target_compile_options( DistributedMeshTest PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( DistributedMeshTest ${GTEST_BOTH_LIBRARIES} )
endif()

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   target_compile_definitions(DistributedMeshTest PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(DistributedMeshTest tinyxml2::tinyxml2)
endif()


SET (mpi_test_parameters_1d -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_1D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_1D COMMAND "mpirun" ${mpi_test_parameters_1d})

SET (mpi_test_parameters_2d -np 9 -H localhost:9 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_2D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_2D COMMAND "mpirun" ${mpi_test_parameters_2d})

SET (mpi_test_parameters_3d -np 27 -H localhost:27 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_3D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_3D COMMAND "mpirun" ${mpi_test_parameters_3d})

# TODO: Fix
#SET (mpi_test_parameters_IO -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridIOTest${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME DistributedGridIOTest COMMAND "mpirun" ${mpi_test_parameters_IO})

# TODO: Fix
#SET (mpi_test_parameters_IOMPIIO -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridIO_MPIIOTest${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME DistributedGridIO_MPIIOTest COMMAND "mpirun" ${mpi_test_parameters_IOMPIIO})

SET (mpi_test_parameters_CutDistributedGridTest -np 12 -H localhost:12 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME CutDistributedGridTest COMMAND "mpirun" ${mpi_test_parameters_CutDistributedGridTest})

# TODO: Fix
#SET (mpi_test_parameters_CutDistributedMeshFunctionTest -np 12 -H localhost:12 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME CutDistributedMeshFunctionTest COMMAND "mpirun" ${mpi_test_parameters_CutDistributedMeshFunctionTest})

# TODO: Fix
#SET (mpi_test_parameters_DistributedVectorFieldIO_MPIIOTest -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorFieldIO_MPIIOTest ${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME DistributedVectorFieldIO_MPIIOTest COMMAND "mpirun" ${mpi_test_parameters_IOMPIIO})

add_test( NAME DistributedMeshTest_2x2 COMMAND mpirun -np 4 -H localhost:4 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )
add_test( NAME DistributedMeshTest_3x3 COMMAND mpirun -np 9 -H localhost:9 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )
add_test( NAME DistributedMeshTest_4x4 COMMAND mpirun -np 16 -H localhost:16 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )

endif()
