ADD_EXECUTABLE( MultiPrecisionTest MultiPrecisionTest.cpp ../MultiPrecision.cpp )
TARGET_COMPILE_OPTIONS( MultiPrecisionTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( MultiPrecisionTest
                           ${GTEST_BOTH_LIBRARIES}
                           ${GMP_LIBRARIES}
                           tnl )
                           
#ADD_EXECUTABLE( QuadTest QuadTest.cpp )
#TARGET_COMPILE_OPTIONS( QuadTest PRIVATE ${CXX_TESTS_FLAGS} )
#TARGET_LINK_LIBRARIES( QuadTest
#                           ${GTEST_BOTH_LIBRARIES}
#                           tnl )
                           
#ADD_EXECUTABLE( DoubleTest DoubleTest.cpp )
#TARGET_COMPILE_OPTIONS( DoubleTest PRIVATE ${CXX_TESTS_FLAGS} )
#TARGET_LINK_LIBRARIES( DoubleTest
#                           ${GTEST_BOTH_LIBRARIES}
#                           tnl )
                           
ADD_TEST( MultiPrecisionTest ${EXECUTABLE_OUTPUT_PATH}/MultiPrecisionTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( QuadTest ${EXECUTABLE_OUTPUT_PATH}/QuadTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( DoubleTest ${EXECUTABLE_OUTPUT_PATH}/DoubleTest${CMAKE_EXECUTABLE_SUFFIX} )
