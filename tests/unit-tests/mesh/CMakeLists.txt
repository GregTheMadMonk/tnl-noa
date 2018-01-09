set( headers tnlGridTester.h
             tnlMeshEntityTester.h
             tnlMeshTester.h )

ADD_EXECUTABLE( tnlGridTest${mpiExt} ${headers} tnlGridTest.cpp )
TARGET_LINK_LIBRARIES( tnlGridTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                       tnl${mpiExt} )

#ADD_EXECUTABLE( tnlMeshEntityTest${mpiExt} ${headers} tnlMeshEntityTest.cpp )
#TARGET_LINK_LIBRARIES( MeshEntityTest${mpiExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt} )

ADD_EXECUTABLE( tnlMeshTest${mpiExt} ${headers} tnlMeshTest.cpp )
TARGET_LINK_LIBRARIES( tnlMeshTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )

                                                              
