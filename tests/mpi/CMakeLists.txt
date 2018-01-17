    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTest ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTest ${CPPUNIT_LIBRARIES}
                                                           tnl )
    TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTest PUBLIC "-DDIMENSION=2" )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTest PUBLIC "-DXDISTR -DYDISTR" )

    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestX ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestX ${CPPUNIT_LIBRARIES}
                                                           tnl )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestX PUBLIC "-DDIMENSION=2" )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestX PUBLIC "-DXDISTR" )

    ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestY ${headers} MeshFunctionEvauateTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestY ${CPPUNIT_LIBRARIES}
                                                           tnl )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestY PUBLIC "-DDIMENSION=2" )
   	TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestY PUBLIC "-DYDISTR" )

    #ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestZ ${headers} MeshFunctionEvauateTest.cpp )   
    #TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestZ ${CPPUNIT_LIBRARIES}
          #                                                  tnl${mpiExt}-0.1 )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestZ PUBLIC "-DDIMENSION=3" )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestZ PUBLIC "-DZDISTR" )

    #ADD_EXECUTABLE( tnlMeshFuncttionEvaluateTestXY ${headers} MeshFunctionEvauateTest.cpp )   
    #TARGET_LINK_LIBRARIES( tnlMeshFuncttionEvaluateTestXY ${CPPUNIT_LIBRARIES}
         #                                                     tnl${mpiExt}-0.1 )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestXY PUBLIC "-DDIMENSION=3" )
	#TARGET_COMPILE_DEFINITIONS( tnlMeshFuncttionEvaluateTestXY PUBLIC "-DXDISTR -DYDISTR" )

