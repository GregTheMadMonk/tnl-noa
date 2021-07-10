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
         ScanTest
         SegmentedScanTest
)
set( CUDA_TESTS
         ScanTestCuda
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
   ADD_EXECUTABLE( DistributedScanTest DistributedScanTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedScanTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedScanTest ${GTEST_BOTH_LIBRARIES} )

   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedScanTestCuda DistributedScanTestCuda.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedScanTestCuda ${GTEST_BOTH_LIBRARIES} )
   endif()

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedScanTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedScanTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )

   if( BUILD_CUDA )
      SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedScanTestCuda COMMAND "mpirun" ${mpi_test_parameters})
      ADD_TEST( NAME DistributedScanTestCuda_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
   endif()
endif()
