    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTest${debugExt} ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTest${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
    #TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTest${debugExt} PUBLIC "-DDIMENSION=2" )
   	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTest${debugExt} PUBLIC "-DXDISTR -DYDISTR" )

    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestX${debugExt} ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestX${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestX${debugExt} PUBLIC "-DDIMENSION=2" )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestX${debugExt} PUBLIC "-DXDISTR" )

    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestY${debugExt} ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestY${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestY${debugExt} PUBLIC "-DDIMENSION=2" )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestY${debugExt} PUBLIC "-DYDISTR" )

    #ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestZ${debugExt} ${headers} MeshFunctionEvauateTest.cpp )   
    #TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestZ${debugExt} ${CPPUNIT_LIBRARIES}
          #                                                  tnl${mpiExt}${debugExt}-0.1 )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestZ${debugExt} PUBLIC "-DDIMENSION=3" )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestZ${debugExt} PUBLIC "-DZDISTR" )

    #ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestXY${debugExt} ${headers} MeshFunctionEvauateTest.cpp )   
    #TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestXY${debugExt} ${CPPUNIT_LIBRARIES}
         #                                                     tnl${mpiExt}${debugExt}-0.1 )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestXY${debugExt} PUBLIC "-DDIMENSION=3" )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestXY${debugExt} PUBLIC "-DXDISTR -DYDISTR" )

