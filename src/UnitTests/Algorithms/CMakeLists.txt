IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( MemoryOperationsTest MemoryOperationsTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MemoryOperationsTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( MultireductionTest MultireductionTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultireductionTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( ParallelForTest ParallelForTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ParallelForTest ${GTEST_BOTH_LIBRARIES} )
ELSE( BUILD_CUDA )
   ADD_EXECUTABLE( MemoryOperationsTest MemoryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( MemoryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MemoryOperationsTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( MultireductionTest MultireductionTest.cpp )
   TARGET_COMPILE_OPTIONS( MultireductionTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultireductionTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( ParallelForTest ParallelForTest.cpp )
   TARGET_COMPILE_OPTIONS( ParallelForTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ParallelForTest ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )


ADD_TEST( MemoryOperationsTest ${EXECUTABLE_OUTPUT_PATH}/MemoryOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MultireductionTest ${EXECUTABLE_OUTPUT_PATH}/MultireductionTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ParallelForTest ${EXECUTABLE_OUTPUT_PATH}/ParallelForTest${CMAKE_EXECUTABLE_SUFFIX} )
