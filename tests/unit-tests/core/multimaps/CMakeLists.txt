set( headers tnlIndexMultimapTester.h )

ADD_EXECUTABLE( tnlEllpackIndexMultimapTest${mpiExt}${debugExt} ${headers} tnlEllpackIndexMultimapTest.cpp )
TARGET_LINK_LIBRARIES( tnlEllpackIndexMultimapTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt} )
