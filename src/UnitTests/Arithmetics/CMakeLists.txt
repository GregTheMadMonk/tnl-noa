

if( HAVE_GMP )
   ADD_EXECUTABLE( MultiPrecisionTest MultiPrecisionTest.cpp ../MultiPrecision.cpp )
   TARGET_COMPILE_OPTIONS( MultiPrecisionTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultiPrecisionTest
                              ${GTEST_BOTH_LIBRARIES}
                              ${GMP_LIBRARIES} )
   ADD_TEST( MultiPrecisionTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MultiPrecisionTest${CMAKE_EXECUTABLE_SUFFIX} )
endif( HAVE_GMP )


# TODO: Fix the following tests                           
#ADD_EXECUTABLE( QuadTest QuadTest.cpp )
#TARGET_COMPILE_OPTIONS( QuadTest PRIVATE ${CXX_TESTS_FLAGS} )
#TARGET_LINK_LIBRARIES( QuadTest ${GTEST_BOTH_LIBRARIES} )
                           
#ADD_EXECUTABLE( DoubleTest DoubleTest.cpp )
#TARGET_COMPILE_OPTIONS( DoubleTest PRIVATE ${CXX_TESTS_FLAGS} )
#TARGET_LINK_LIBRARIES( DoubleTest ${GTEST_BOTH_LIBRARIES} )
                           

#ADD_TEST( QuadTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/QuadTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( DoubleTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DoubleTest${CMAKE_EXECUTABLE_SUFFIX} )
