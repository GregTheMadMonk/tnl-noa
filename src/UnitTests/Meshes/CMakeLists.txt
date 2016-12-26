ADD_EXECUTABLE( BoundaryTagsTest${mpiExt}${debugExt} ${headers} BoundaryTagsTest.cpp )
SET_TARGET_PROPERTIES( BoundaryTagsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( BoundaryTagsTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( MeshTest${mpiExt}${debugExt} ${headers} MeshTest.cpp )
SET_TARGET_PROPERTIES( MeshTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( MeshTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( MeshEntityTest${mpiExt}${debugExt} ${headers} MeshEntityTest.cpp )
SET_TARGET_PROPERTIES( MeshEntityTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( MeshEntityTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( MultimapTest${mpiExt}${debugExt} ${headers} MultimapTest.cpp )
SET_TARGET_PROPERTIES( MultimapTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( MultimapTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( StaticMultimapTest${mpiExt}${debugExt} ${headers} StaticMultimapTest.cpp )
SET_TARGET_PROPERTIES( StaticMultimapTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( StaticMultimapTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )


ADD_TEST( BoundaryTagsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/BoundaryTagsTest${mpiExt}${debugExt} )
ADD_TEST( MeshTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshTest${mpiExt}${debugExt} )
ADD_TEST( MeshEntityTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${mpiExt}${debugExt} )
ADD_TEST( MultimapTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${mpiExt}${debugExt} )
ADD_TEST( StaticMultimapTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${mpiExt}${debugExt} )
