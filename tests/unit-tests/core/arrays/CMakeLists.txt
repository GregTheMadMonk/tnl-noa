set( headers tnlMultiArrayTester.h
             tnlArrayTester.h
             tnlSharedArrayTester.h )

ADD_EXECUTABLE( tnlArrayTest${mpiExt}${debugExt} ${headers} tnlArrayTest.cpp )
TARGET_LINK_LIBRARIES( tnlArrayTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
             
if( BUILD_CUDA )
endif()
                                                                       