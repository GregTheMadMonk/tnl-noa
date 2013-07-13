ADD_SUBDIRECTORY( cuda )

set( headers tnlFileTester.h
             tnlStringTester.h
             tnlObjectTester.h
             tnlRealTester.h
             tnlTupleTester.h
             tnlVectorTester.h
             tnlMultiArrayTester.h
             tnlArrayTester.h
             tnlSharedArrayTester.h
             tnlSharedVectorTester.h
             tnlGridOldTester.h
             tnlSharedMemoryTester.h
             tnlCommunicatorTester.h )

ADD_EXECUTABLE( tnlArrayTest${mpiExt}${debugExt} ${headers} tnlArrayTest.cpp )
TARGET_LINK_LIBRARIES( tnlArrayTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
             
if( BUILD_CUDA )
endif()
                                                                       