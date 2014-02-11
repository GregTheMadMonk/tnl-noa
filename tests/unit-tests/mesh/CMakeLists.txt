set( headers tnlMeshEntityTester.h )

ADD_EXECUTABLE( tnlMeshEntityTest${mpiExt}${debugExt} ${headers} tnlMeshEntityTest.cpp )
TARGET_LINK_LIBRARIES( tnlMeshEntityTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

