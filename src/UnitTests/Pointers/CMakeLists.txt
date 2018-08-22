ADD_EXECUTABLE( UniquePointerTest UniquePointerTest.cpp )
TARGET_COMPILE_OPTIONS( UniquePointerTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( UniquePointerTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )
ADD_TEST( UniquePointerTest ${EXECUTABLE_OUTPUT_PATH}/UniquePointerTest${CMAKE_EXECUTABLE_SUFFIX} )
