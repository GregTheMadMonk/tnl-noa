ADD_EXECUTABLE( ListTest${mpiExt}${debugExt} ListTest.cpp )
TARGET_COMPILE_OPTIONS( ListTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ListTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${debugExt} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayOperationsTest${mpiExt}${debugExt} ArrayOperationsTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayOperationsTest${mpiExt}${debugExt} ArrayOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( ArrayOperationsTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayTest${mpiExt}${debugExt} ArrayTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayTest${mpiExt}${debugExt} ArrayTest.cpp )
   TARGET_COMPILE_OPTIONS( ArrayTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( ArrayTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticArrayTest${mpiExt}${debugExt} StaticArrayTest.cpp )
TARGET_COMPILE_OPTIONS( StaticArrayTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticArrayTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${debugExt} )

# NOTE: Vector = Array + VectorOperations, so we test Vector and VectorOperations at the same time
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( VectorTest${mpiExt}${debugExt} VectorTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( VectorTest${mpiExt}${debugExt} VectorTest.cpp )
   TARGET_COMPILE_OPTIONS( VectorTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( VectorTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticVectorTest${mpiExt}${debugExt} StaticVectorTest.cpp )
TARGET_COMPILE_OPTIONS( StaticVectorTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticVectorTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${debugExt} )

#IF( BUILD_CUDA )
#   CUDA_ADD_EXECUTABLE( MultiArrayTest${mpiExt}${debugExt} MultiArrayTest.cu
#                        OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MultiArrayTest${mpiExt}${debugExt}
#                              ${GTEST_BOTH_LIBRARIES}
#                              tnl${debugExt} )
#ELSE(  BUILD_CUDA )
#   ADD_EXECUTABLE( MultiArrayTest${mpiExt}${debugExt} MultiArrayTest.cpp )
#   TARGET_COMPILE_OPTIONS( MultiArrayTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MultiArrayTest${mpiExt}${debugExt}
#                              ${GTEST_BOTH_LIBRARIES}
#                              tnl${debugExt} )
#ENDIF( BUILD_CUDA )


ADD_TEST( ListTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ListTest${mpiExt}${debugExt} )
ADD_TEST( ArrayOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ArrayOperationsTest${mpiExt}${debugExt} )
ADD_TEST( ArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ArrayTest${mpiExt}${debugExt} )
ADD_TEST( StaticArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StaticArrayTest${mpiExt}${debugExt} )
ADD_TEST( VectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/VectorTest${mpiExt}${debugExt} )
ADD_TEST( StaticVectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StaticVectorTest${mpiExt}${debugExt} )
#ADD_TEST( MultiArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultiArrayTest${mpiExt}${debugExt} )
