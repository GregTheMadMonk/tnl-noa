set( headers tnlMeshEntityTester.h
             tnlMeshTester.h )

#ADD_EXECUTABLE( tnlMeshEntityTest${mpiExt}${debugExt} ${headers} tnlMeshEntityTest.cpp )
#TARGET_LINK_LIBRARIES( tnlMeshEntityTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt}${debugExt}-0.1 )

#ADD_EXECUTABLE( tnlMeshTest${mpiExt}${debugExt} ${headers} tnlMeshTest.cpp )
#TARGET_LINK_LIBRARIES( tnlMeshTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt}${debugExt}-0.1 )

                                                              