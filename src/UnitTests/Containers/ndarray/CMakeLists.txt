add_executable( NDArrayTest NDArrayTest.cpp )
target_compile_options( NDArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
target_link_libraries( NDArrayTest ${GTEST_BOTH_LIBRARIES} )
add_test( NDArrayTest ${EXECUTABLE_OUTPUT_PATH}/NDArrayTest${CMAKE_EXECUTABLE_SUFFIX} )

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
