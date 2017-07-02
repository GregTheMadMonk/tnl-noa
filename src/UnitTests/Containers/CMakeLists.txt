IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayOperationsTest${mpiExt}${debugExt} ArrayOperationsTest.h ArrayOperationsTest.cu )
   SET_TARGET_PROPERTIES( ArrayOperationsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayOperationsTest${mpiExt}${debugExt} ArrayOperationsTest.h ArrayOperationsTest.cpp )
   SET_TARGET_PROPERTIES( ArrayOperationsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayOperationsTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayTest${mpiExt}${debugExt} ArrayTest.h ArrayTest.cu )
   SET_TARGET_PROPERTIES( ArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( ArrayTest${mpiExt}${debugExt} ArrayTest.h ArrayTest.cpp )
   SET_TARGET_PROPERTIES( ArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( ArrayTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( MultiArrayTest${mpiExt}${debugExt} MultiArrayTest.h MultiArrayTest.cu )
   SET_TARGET_PROPERTIES( MultiArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( MultiArrayTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( MultiArrayTest${mpiExt}${debugExt} MultiArrayTest.h MultiArrayTest.cpp )
   SET_TARGET_PROPERTIES( MultiArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( MultiArrayTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( StaticArrayTest${mpiExt}${debugExt} StaticArrayTest.h StaticArrayTest.cu )
   SET_TARGET_PROPERTIES( StaticArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( StaticArrayTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( StaticArrayTest${mpiExt}${debugExt} StaticArrayTest.h StaticArrayTest.cpp )
   SET_TARGET_PROPERTIES( StaticArrayTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( StaticArrayTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( VectorOperationsTest${mpiExt}${debugExt} VectorOperationsTest.h VectorOperationsTest.cu )
   SET_TARGET_PROPERTIES( VectorOperationsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( VectorOperationsTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( VectorOperationsTest${mpiExt}${debugExt} VectorOperationsTest.h VectorOperationsTest.cpp )
   SET_TARGET_PROPERTIES( VectorOperationsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( VectorOperationsTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( StaticVectorTest${mpiExt}${debugExt} StaticVectorTest.h StaticVectorTest.cu )
   SET_TARGET_PROPERTIES( StaticVectorTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( StaticVectorTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( StaticVectorTest${mpiExt}${debugExt} StaticVectorTest.h StaticVectorTest.cpp )
   SET_TARGET_PROPERTIES( StaticVectorTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( StaticVectorTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


ADD_TEST( ArrayOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ArrayOperationsTest${mpiExt}${debugExt} )
ADD_TEST( ArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ArrayTest${mpiExt}${debugExt} )
ADD_TEST( MultiArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultiArrayTest${mpiExt}${debugExt} )
ADD_TEST( StaticArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StaticArrayTest${mpiExt}${debugExt} )
ADD_TEST( VectorOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/VectorOperationsTest${mpiExt}${debugExt} )
ADD_TEST( StaticVectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StaticVectorTest${mpiExt}${debugExt} )
