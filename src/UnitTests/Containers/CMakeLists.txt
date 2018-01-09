ADD_EXECUTABLE( ListTest ListTest.cpp )
TARGET_COMPILE_OPTIONS( ListTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ListTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayOperationsTest ArrayOperationsTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayOperationsTest ArrayOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( ArrayOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayTest ArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayTest ArrayTest.cpp )
   TARGET_COMPILE_OPTIONS( ArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticArrayTest StaticArrayTest.cpp )
TARGET_COMPILE_OPTIONS( StaticArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticArrayTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

# NOTE: Vector = Array + VectorOperations, so we test Vector and VectorOperations at the same time
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( VectorTest VectorTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( VectorTest VectorTest.cpp )
   TARGET_COMPILE_OPTIONS( VectorTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticVectorTest StaticVectorTest.cpp )
TARGET_COMPILE_OPTIONS( StaticVectorTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticVectorTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

#IF( BUILD_CUDA )
#   CUDA_ADD_EXECUTABLE( MultiArrayTest MultiArrayTest.cu
#                        OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MultiArrayTest
#                              ${GTEST_BOTH_LIBRARIES}
#                              tnl )
#ELSE(  BUILD_CUDA )
#   ADD_EXECUTABLE( MultiArrayTest MultiArrayTest.cpp )
#   TARGET_COMPILE_OPTIONS( MultiArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MultiArrayTest
#                              ${GTEST_BOTH_LIBRARIES}
#                              tnl )
#ENDIF( BUILD_CUDA )


ADD_TEST( ListTest ${EXECUTABLE_OUTPUT_PATH}/ListTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ArrayOperationsTest ${EXECUTABLE_OUTPUT_PATH}/ArrayOperationsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ArrayTest ${EXECUTABLE_OUTPUT_PATH}/ArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticArrayTest ${EXECUTABLE_OUTPUT_PATH}/StaticArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( VectorTest ${EXECUTABLE_OUTPUT_PATH}/VectorTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticVectorTest ${EXECUTABLE_OUTPUT_PATH}/StaticVectorTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( MultiArrayTest ${EXECUTABLE_OUTPUT_PATH}/MultiArrayTest${CMAKE_EXECUTABLE_SUFFIX} )
