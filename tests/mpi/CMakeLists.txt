ADD_EXECUTABLE( MeshFunctionEvaluateTest ${headers} MeshFunctionEvaluateTest.cpp )
TARGET_LINK_LIBRARIES( MeshFunctionEvaluateTest ${CPPUNIT_LIBRARIES} tnl )
TARGET_COMPILE_DEFINITIONS( MeshFunctionEvaluateTest PUBLIC "-DDIMENSION=2" )
TARGET_COMPILE_DEFINITIONS( MeshFunctionEvaluateTest PUBLIC "-DXDISTR -DYDISTR" )

ADD_EXECUTABLE( MeshFunctionEvaluateTestX ${headers} MeshFunctionEvaluateTest.cpp )
TARGET_LINK_LIBRARIES( MeshFunctionEvaluateTestX ${CPPUNIT_LIBRARIES} tnl )
TARGET_COMPILE_DEFINITIONS( MeshFunctionEvaluateTestX PUBLIC "-DDIMENSION=2" )
TARGET_COMPILE_DEFINITIONS( MeshFunctionEvaluateTestX PUBLIC "-DXDISTR" )

ADD_EXECUTABLE( MeshFunctionEvaluateTestY ${headers} MeshFunctionEvaluateTest.cpp )
TARGET_LINK_LIBRARIES( MeshFunctionEvaluateTestY ${CPPUNIT_LIBRARIES} tnl )
TARGET_COMPILE_DEFINITIONS( MeshFunctionEvaluateTestY PUBLIC "-DDIMENSION=2" )
TARGET_COMPILE_DEFINITIONS( MeshFunctionEvaluateTestY PUBLIC "-DYDISTR" )


ADD_EXECUTABLE( mpiio-save-test ${headers} mpiio-save-test.cpp )
TARGET_LINK_LIBRARIES(  mpiio-save-test ${CPPUNIT_LIBRARIES} tnl )

ADD_EXECUTABLE( mpiio-save-load-test ${headers} mpiio-save-load-test.cpp )
TARGET_LINK_LIBRARIES(  mpiio-save-load-test ${CPPUNIT_LIBRARIES} tnl )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( mpi-gpu-test ${headers} mpi-gpu.cu )
   TARGET_LINK_LIBRARIES( mpi-gpu-test ${CPPUNIT_LIBRARIES} tnl )
   TARGET_COMPILE_DEFINITIONS( mpi-gpu-test PUBLIC ${MIC_CXX_FLAGS} )

   CUDA_ADD_EXECUTABLE( GPUmeshFunctionEvaluateTest ${headers} GPUmeshFunctionEvaluateTest.cu )
   TARGET_LINK_LIBRARIES( GPUmeshFunctionEvaluateTest ${CPPUNIT_LIBRARIES} tnl )
   TARGET_COMPILE_DEFINITIONS( GPUmeshFunctionEvaluateTest PUBLIC ${MIC_CXX_FLAGS} )

   CUDA_ADD_EXECUTABLE( mpiio-save-test-gpu ${headers} mpiio-save-test.cu )
   TARGET_LINK_LIBRARIES(  mpiio-save-test-gpu ${CPPUNIT_LIBRARIES} tnl )
   TARGET_COMPILE_DEFINITIONS( mpiio-save-test-gpu PUBLIC ${MIC_CXX_FLAGS} )
ENDIF( BUILD_CUDA )
