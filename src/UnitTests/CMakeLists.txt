#ADD_SUBDIRECTORY( Containers )

ADD_EXECUTABLE( UniquePointerTest${mpiExt}${debugExt} ${headers} UniquePointerTest.cpp )
SET_TARGET_PROPERTIES( UniquePointerTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( UniquePointerTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( FileTest${mpiExt}${debugExt} FileTest.h FileTest.cu )
   SET_TARGET_PROPERTIES( FileTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( FileTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE( FileTest${mpiExt}${debugExt} FileTest.h FileTest.cpp )
   SET_TARGET_PROPERTIES( FileTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( FileTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StringTest${mpiExt}${debugExt} ${headers} StringTest.cpp )
SET_TARGET_PROPERTIES( StringTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( StringTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( ListTest${mpiExt}${debugExt} ${headers} ListTest.cpp )
SET_TARGET_PROPERTIES( ListTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( ListTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( ObjectTest${mpiExt}${debugExt} ${headers} ObjectTest.cpp )
SET_TARGET_PROPERTIES( ObjectTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( ObjectTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES} 
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )


ADD_TEST( FileTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/FileTest${mpiExt}${debugExt} )
ADD_TEST( StringTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StringTest${mpiExt}${debugExt} )
ADD_TEST( ListTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ListTest${mpiExt}${debugExt} )
ADD_TEST( ObjectTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ObjectTest${mpiExt}${debugExt} )
ADD_TEST( UniquePointerTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/UniquePointerTest${mpiExt}${debugExt} )
