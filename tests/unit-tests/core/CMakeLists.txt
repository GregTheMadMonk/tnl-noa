ADD_SUBDIRECTORY( arrays )
ADD_SUBDIRECTORY( cuda )
ADD_SUBDIRECTORY( vectors )

set( headers tnlFileTester.h
             tnlStringTester.h
             tnlObjectTester.h
             tnlRealTester.h
             tnlTupleTester.h
             tnlGridOldTester.h
             tnlSharedMemoryTester.h
             tnlCommunicatorTester.h )

ADD_EXECUTABLE( tnlStringTest${mpiExt}${debugExt} ${headers} tnlStringTest.cpp )
TARGET_LINK_LIBRARIES( tnlStringTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE( tnlObjectTest${mpiExt}${debugExt} ${headers} tnlObjectTest.cpp )
TARGET_LINK_LIBRARIES( tnlObjectTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
                                                           
                                                                       