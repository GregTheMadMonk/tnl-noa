    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTest${debugExt} ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTest${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )

