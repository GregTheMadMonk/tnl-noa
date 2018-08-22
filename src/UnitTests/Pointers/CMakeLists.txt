ADD_EXECUTABLE( UniquePointerTest UniquePointerTest.cpp )
TARGET_COMPILE_OPTIONS( UniquePointerTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( UniquePointerTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )
ADD_TEST( UniquePointerTest ${EXECUTABLE_OUTPUT_PATH}/UniquePointerTest${CMAKE_EXECUTABLE_SUFFIX} )


ADD_EXECUTABLE( SharedPointerHostTest SharedPointerHostTest.cpp )
TARGET_COMPILE_OPTIONS( SharedPointerHostTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( SharedPointerHostTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )
ADD_TEST( SharedPointerHostTest ${EXECUTABLE_OUTPUT_PATH}/SharedPointerHostTest${CMAKE_EXECUTABLE_SUFFIX} )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SharedPointerCudaTest SharedPointerCudaTest.cu 
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SharedPointerCudaTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
   ADD_TEST( SharedPointerCudaTest ${EXECUTABLE_OUTPUT_PATH}/SharedPointerCudaTest${CMAKE_EXECUTABLE_SUFFIX} )
endif( BUILD_CUDA )
