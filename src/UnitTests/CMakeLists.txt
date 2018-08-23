ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Matrices )
ADD_SUBDIRECTORY( Meshes )
ADD_SUBDIRECTORY( Pointers )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( FileTest FileTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( FileTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE( FileTest FileTest.cpp )
   TARGET_COMPILE_OPTIONS( FileTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( FileTest
                              ${GTEST_BOTH_LIBRARIES}
                              tnl )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StringTest StringTest.cpp )
TARGET_COMPILE_OPTIONS( StringTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StringTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

ADD_EXECUTABLE( ObjectTest ObjectTest.cpp )
TARGET_COMPILE_OPTIONS( ObjectTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ObjectTest
                           ${GTEST_BOTH_LIBRARIES} 
                           tnl )

ADD_TEST( FileTest ${EXECUTABLE_OUTPUT_PATH}/FileTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StringTest ${EXECUTABLE_OUTPUT_PATH}/StringTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( ObjectTest ${EXECUTABLE_OUTPUT_PATH}/ObjectTest${CMAKE_EXECUTABLE_SUFFIX} )

