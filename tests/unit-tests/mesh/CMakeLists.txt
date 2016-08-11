set( headers tnlGridTester.h
             tnlMeshEntityTester.h
             tnlMeshTester.h )

ADD_EXECUTABLE( tnlGridTest${mpiExt}${debugExt} ${headers} tnlGridTest.cpp )
TARGET_LINK_LIBRARIES( tnlGridTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                       tnl${mpiExt}${debugExt}-0.1 )

#ADD_EXECUTABLE( tnlMeshEntityTest${mpiExt}${debugExt} ${headers} tnlMeshEntityTest.cpp )
#TARGET_LINK_LIBRARIES( MeshEntityTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE( tnlMeshTest${mpiExt}${debugExt} ${headers} tnlMeshTest.cpp )
TARGET_LINK_LIBRARIES( tnlMeshTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

                                                              