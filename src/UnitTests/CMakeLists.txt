if( WITH_TESTS STREQUAL "yes" )

ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Matrices )

ADD_EXECUTABLE( UniquePointerTest${mpiExt}${debugExt} UniquePointerTest.cpp )
TARGET_COMPILE_OPTIONS( UniquePointerTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( UniquePointerTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${debugExt} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( FileTest${mpiExt}${debugExt} FileTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( FileTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE( FileTest${mpiExt}${debugExt} FileTest.cpp )
   TARGET_COMPILE_OPTIONS( FileTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( FileTest${mpiExt}${debugExt}
                              ${GTEST_BOTH_LIBRARIES}
                              tnl${debugExt} )
ENDIF( BUILD_CUDA )

ADD_EXECUTABLE( StringTest${mpiExt}${debugExt} StringTest.cpp )
TARGET_COMPILE_OPTIONS( StringTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StringTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${debugExt} )

ADD_EXECUTABLE( ObjectTest${mpiExt}${debugExt} ObjectTest.cpp )
TARGET_COMPILE_OPTIONS( ObjectTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( ObjectTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES} 
                           tnl${debugExt} )


ADD_TEST( FileTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/FileTest${mpiExt}${debugExt} )
ADD_TEST( StringTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/StringTest${mpiExt}${debugExt} )
ADD_TEST( ObjectTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/ObjectTest${mpiExt}${debugExt} )
ADD_TEST( UniquePointerTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/UniquePointerTest${mpiExt}${debugExt} )

endif( WITH_TESTS STREQUAL "yes" )
