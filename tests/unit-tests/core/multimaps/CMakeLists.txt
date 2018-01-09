set( headers tnlIndexMultimapTester.h )

ADD_EXECUTABLE( tnlEllpackIndexMultimapTest${mpiExt} ${headers} tnlEllpackIndexMultimapTest.cpp )
TARGET_LINK_LIBRARIES( tnlEllpackIndexMultimapTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )
