set( headers core/tnl-cuda-kernels.h 
             core/tnlMultiArrayTester.h 
             core/tnlCUDAKernelsTester.h 
             core/tnlCommunicatorTester.h 
             core/tnlFileTester.h 
             core/tnlGridOldTester.h
             core/tnlObjectTester.h  
             core/tnlVectorTester.h 
             core/tnlRealTester.h 
             core/tnlSharedMemoryTester.h
             core/tnlStringTester.h 
             core/tnlTupleTester.h 
             diff/tnlMPIMeshTester.h 
             diff/tnlMersonSolverCUDATester.cu.h
             matrix/tnlMatrixTester.h
             matrix/tnlCSRMatrixTester.h
             matrix/tnlEllpackMatrixTester.h
             matrix/tnlRgCSRMatrixTester.h 
             matrix/tnlAdaptiveRgCSRMatrixTester.h
             solver/tnlMersonSolverTester.h )
             
if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-unit-tests${mpiExt}${debugExt} ${headers} tnl-unit-tests.cu )
    TARGET_LINK_LIBRARIES( tnl-unit-tests${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
else()
    ADD_EXECUTABLE( tnl-unit-tests${mpiExt}${debugExt} ${headers} tnl-unit-tests.cpp )
    TARGET_LINK_LIBRARIES( tnl-unit-tests${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
endif()

ADD_TEST( tnl-unit-tests${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-unit-tests${mpiExt}${debugExt}  )                                                                       
                                                                       