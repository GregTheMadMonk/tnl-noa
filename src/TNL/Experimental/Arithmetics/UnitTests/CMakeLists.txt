ADD_EXECUTABLE( QuadTest${mpiExt}${debugExt} QuadTest.cpp )
TARGET_COMPILE_OPTIONS( QuadTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( QuadTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )
                           
ADD_EXECUTABLE( MultiPrecisionTest${mpiExt}${debugExt} MultiPrecisionTest.cpp )
TARGET_COMPILE_OPTIONS( MultiPrecisionTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( MultiPrecisionTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           ${GMP_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )
                           
ADD_TEST( QuadTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/QuadTest${mpiExt}${debugExt} )
ADD_TEST( MultiPrecisionTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultiPrecisionTest${mpiExt}${debugExt} )