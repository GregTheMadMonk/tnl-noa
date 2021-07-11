ADD_SUBDIRECTORY( Segments )
ADD_SUBDIRECTORY( Sorting )

set( COMMON_TESTS
         MemoryOperationsTest
         MultireductionTest
         ParallelForTest
         reduceTest
         staticForTest
         unrolledForTest
)

set( CPP_TESTS
         scanTest
         SegmentedScanTest
)
set( CUDA_TESTS
         scanTestCuda
)
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPP_TESTS  ${CPP_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${GTEST_BOTH_LIBRARIES} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      cuda_add_executable( ${target} ${target}.cu OPTIONS ${CXX_TESTS_FLAGS} )
      target_link_libraries( ${target} ${GTEST_BOTH_LIBRARIES} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()


if( ${BUILD_MPI} )
   ADD_EXECUTABLE( distributedScanTest distributedScanTest.cpp )
   TARGET_COMPILE_OPTIONS( distributedScanTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( distributedScanTest ${GTEST_BOTH_LIBRARIES} )

   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( distributedScanTestCuda distributedScanTestCuda.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( distributedScanTestCuda ${GTEST_BOTH_LIBRARIES} )
   endif()

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME distributedScanTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME distributedScanTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )

   if( BUILD_CUDA )
      SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME distributedScanTestCuda COMMAND "mpirun" ${mpi_test_parameters})
      ADD_TEST( NAME distributedScanTestCuda_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
   endif()
endif()
