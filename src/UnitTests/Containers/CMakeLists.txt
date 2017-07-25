ADD_EXECUTABLE( ListTest${mpiExt}${debugExt} ListTest.cpp )
SET_TARGET_PROPERTIES( ListTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( ListTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayOperationsTest${mpiExt}${debugExt} ArrayOperationsTest.cu
                        OPTIONS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayOperationsTest${mpiExt}${debugExt} ArrayOperationsTest.cpp )
   SET_TARGET_PROPERTIES( ArrayOperationsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayTest${mpiExt}${debugExt} ArrayTest.cu
                        OPTIONS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayTest${mpiExt}${debugExt} ArrayTest.cpp )
   SET_TARGET_PROPERTIES( ArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticArrayTest${mpiExt}${debugExt} StaticArrayTest.cpp )
SET_TARGET_PROPERTIES( StaticArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( StaticArrayTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

# NOTE: Vector = Array + VectorOperations, so we test Vector and VectorOperations at the same time
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( VectorTest${mpiExt}${debugExt} VectorTest.cu
                        OPTIONS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( VectorTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( VectorTest${mpiExt}${debugExt} VectorTest.cpp )
   SET_TARGET_PROPERTIES( VectorTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( VectorTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StaticVectorTest${mpiExt}${debugExt} StaticVectorTest.cpp )
SET_TARGET_PROPERTIES( StaticVectorTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( StaticVectorTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

#IF( BUILD_CUDA )
#   CUDA_ADD_EXECUTABLE( MultiArrayTest${mpiExt}${debugExt} MultiArrayTest.cu
#                        OPTIONS "${CXX_TESTS_FLAGS}" )
#   TARGET_LINK_LIBRARIES( MultiArrayTest${mpiExt}${debugExt}
#                              ${GTEST_BOTH_LIBRARIES}
#                              tnl${mpiExt}${debugExt}-${tnlVersion} )
#ELSE(  BUILD_CUDA )
#   ADD_EXECUTABLE( MultiArrayTest${mpiExt}${debugExt} MultiArrayTest.cpp )
#   SET_TARGET_PROPERTIES( MultiArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
#   TARGET_LINK_LIBRARIES( MultiArrayTest${mpiExt}${debugExt}
#                              ${GTEST_BOTH_LIBRARIES}
#                              tnl${mpiExt}${debugExt}-${tnlVersion} )
#ENDIF( BUILD_CUDA )


ADD_TEST( ListTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ListTest${mpiExt}${debugExt} )
ADD_TEST( ArrayOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ArrayOperationsTest${mpiExt}${debugExt} )
ADD_TEST( ArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ArrayTest${mpiExt}${debugExt} )
ADD_TEST( StaticArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StaticArrayTest${mpiExt}${debugExt} )
ADD_TEST( VectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/VectorTest${mpiExt}${debugExt} )
ADD_TEST( StaticVectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StaticVectorTest${mpiExt}${debugExt} )
#ADD_TEST( MultiArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultiArrayTest${mpiExt}${debugExt} )
