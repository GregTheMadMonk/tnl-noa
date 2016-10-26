#ADD_EXECUTABLE( UniquePointerTest${mpiExt}${debugExt} ${headers} UniquePointerTest.cpp )
#SET_TARGET_PROPERTIES( UniquePointerTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
#TARGET_LINK_LIBRARIES( UniquePointerTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
#                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
#
#ADD_TEST( FileTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/FileTest${mpiExt}${debugExt} )
