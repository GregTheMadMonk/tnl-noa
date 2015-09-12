set( headers tnlNetworkTester.h )

ADD_EXECUTABLE( tnlEllpackNetworkTest${mpiExt}${debugExt} ${headers} tnlEllpackNetworkTest.cpp )
TARGET_LINK_LIBRARIES( tnlEllpackNetworkTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )