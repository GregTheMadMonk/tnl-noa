ADD_EXECUTABLE( QuadTest${mpiExt}${debugExt} ${headers} QuadTest.cpp )
SET_TARGET_PROPERTIES( QuadTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( QuadTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                        tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( MultiPrecisionTest${mpiExt}${debugExt} ${headers} MultiPrecisionTest.cpp )
SET_TARGET_PROPERTIES( MultiPrecisionTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( MultiPrecisionTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                        tnl${mpiExt}${debugExt}-${tnlVersion} )
                                                        
ADD_TEST( QuadTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/QuadTest${mpiExt}${debugExt} )
ADD_TEST( MultiPrecisionTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultiPrecisionTest${mpiExt}${debugExt} )