ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( solver )

set( headers tnlUnitTestStarter.h
             core/tnl-cuda-kernels.h  
             core/tnlCUDAKernelsTester.h 
             core/tnlCommunicatorTester.h 
             core/tnlFileTester.h 
             core/tnlGridOldTester.h
             core/tnlObjectTester.h   
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
ADD_TEST( core/arrays/tnlArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest${mpiExt}${debugExt} )
ADD_TEST( core/vectors/tnlVectorOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest${mpiExt}${debugExt} )

if( BUILD_CUDA )
   ADD_TEST( core/cuda/tnl-device-check-test${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-device-check-test${mpiExt}${debugExt} )
   ADD_TEST( core/cuda/tnl-memory-operations-test${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-memory-operations-test${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/cuda/tnl-memory-operations-test${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnl-device-check-test${mpiExt}${debugExt} )
   ADD_TEST( core/cuda/tnl-reduction-test${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-reduction-test${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/cuda/tnl-reduction-test${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnl-memory-operations-test${mpiExt}${debugExt} )   
   ADD_TEST( core/vectors/tnlCudaVectorOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCudaVectorOperationsTest${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlCudaVectorOperationsTest${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnl-reduction-test${mpiExt}${debugExt} )         
endif()

ADD_TEST( tnl-unit-tests${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-unit-tests${mpiExt}${debugExt}  )                                                                       
                                                                       