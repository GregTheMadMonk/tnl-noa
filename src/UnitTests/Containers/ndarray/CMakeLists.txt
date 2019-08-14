if( BUILD_CUDA )
   cuda_add_executable( NDArrayTest NDArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   target_link_libraries( NDArrayTest ${GTEST_BOTH_LIBRARIES} )
   add_test( NDArrayTest ${EXECUTABLE_OUTPUT_PATH}/NDArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
else()
   add_executable( NDArrayTest NDArrayTest.cpp )
   target_compile_options( NDArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( NDArrayTest ${GTEST_BOTH_LIBRARIES} )
   add_test( NDArrayTest ${EXECUTABLE_OUTPUT_PATH}/NDArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
endif()

add_executable( NDSubarrayTest NDSubarrayTest.cpp )
target_compile_options( NDSubarrayTest PRIVATE ${CXX_TESTS_FLAGS} )
target_link_libraries( NDSubarrayTest ${GTEST_BOTH_LIBRARIES} )
add_test( NDSubarrayTest ${EXECUTABLE_OUTPUT_PATH}/NDSubarrayTest${CMAKE_EXECUTABLE_SUFFIX} )

add_executable( SlicedNDArrayTest SlicedNDArrayTest.cpp )
target_compile_options( SlicedNDArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
target_link_libraries( SlicedNDArrayTest ${GTEST_BOTH_LIBRARIES} )
add_test( SlicedNDArrayTest ${EXECUTABLE_OUTPUT_PATH}/SlicedNDArrayTest${CMAKE_EXECUTABLE_SUFFIX} )

add_executable( StaticNDArrayTest StaticNDArrayTest.cpp )
target_compile_options( StaticNDArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
target_link_libraries( StaticNDArrayTest ${GTEST_BOTH_LIBRARIES} )
add_test( StaticNDArrayTest ${EXECUTABLE_OUTPUT_PATH}/StaticNDArrayTest${CMAKE_EXECUTABLE_SUFFIX} )

if( BUILD_CUDA )
   cuda_add_executable( StaticNDArrayCudaTest StaticNDArrayCudaTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   target_link_libraries( StaticNDArrayCudaTest ${GTEST_BOTH_LIBRARIES} )
   add_test( StaticNDArrayCudaTest ${EXECUTABLE_OUTPUT_PATH}/StaticNDArrayCudaTest${CMAKE_EXECUTABLE_SUFFIX} )
endif()

if( ${BUILD_MPI} )
   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedNDArray_1D_test DistributedNDArray_1D_test.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_1D_test ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedNDArray_semi1D_test DistributedNDArray_semi1D_test.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_semi1D_test ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedNDArrayOverlaps_1D_test DistributedNDArrayOverlaps_1D_test.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_1D_test ${GTEST_BOTH_LIBRARIES} )

      CUDA_ADD_EXECUTABLE( DistributedNDArrayOverlaps_semi1D_test DistributedNDArrayOverlaps_semi1D_test.cu
                           OPTIONS ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_semi1D_test ${GTEST_BOTH_LIBRARIES} )
   else()
      ADD_EXECUTABLE( DistributedNDArray_1D_test DistributedNDArray_1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArray_1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_1D_test ${GTEST_BOTH_LIBRARIES} )

      ADD_EXECUTABLE( DistributedNDArray_semi1D_test DistributedNDArray_semi1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArray_semi1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_semi1D_test ${GTEST_BOTH_LIBRARIES} )

      ADD_EXECUTABLE( DistributedNDArrayOverlaps_1D_test DistributedNDArrayOverlaps_1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArrayOverlaps_1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_1D_test ${GTEST_BOTH_LIBRARIES} )

      ADD_EXECUTABLE( DistributedNDArrayOverlaps_semi1D_test DistributedNDArrayOverlaps_semi1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArrayOverlaps_semi1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_semi1D_test ${GTEST_BOTH_LIBRARIES} )
   endif()

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedNDArray_1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArray_1D_test COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedNDArray_semi1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArray_semi1D_test COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedNDArrayOverlaps_1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArrayOverlaps_1D_test COMMAND "mpirun" ${mpi_test_parameters})

   SET( mpi_test_parameters -np 4 -H localhost:4 "${EXECUTABLE_OUTPUT_PATH}/DistributedNDArrayOverlaps_semi1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArrayOverlaps_semi1D_test COMMAND "mpirun" ${mpi_test_parameters})
endif()
