ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( matrices )
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

ADD_TEST( core/tnlStringTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlStringTest${mpiExt}${debugExt} )
ADD_TEST( core/tnlObjectTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlObjectTest${mpiExt}${debugExt} )
ADD_TEST( core/arrays/tnlArrayOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayOperationsTest${mpiExt}${debugExt} )
ADD_TEST( core/arrays/tnlArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest${mpiExt}${debugExt} )
ADD_TEST( core/vectors/tnlVectorOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest${mpiExt}${debugExt} )
ADD_TEST( core/vectors/tnlVectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorTest${mpiExt}${debugExt} )

ADD_TEST( matrices/tnlDenseMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlTridiagonalMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlMultidiagonalMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlEllpackMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlSlicedEllpackMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlChunkedEllpackMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlCSRMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRMatrixTest${mpiExt}${debugExt} )

if( BUILD_CUDA )
   ADD_TEST( core/cuda/tnlCudaTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCudaTest${mpiExt}${debugExt} )
   
   ADD_TEST( core/cuda/tnl-reduction-test${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-reduction-test${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/cuda/tnl-reduction-test${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlCudaTest${mpiExt}${debugExt} )   
   
   ADD_TEST( core/arrays/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnl-reduction-test${mpiExt}${debugExt} )
   
   ADD_TEST( core/arrays/tnlArrayTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} )   
   
   ADD_TEST( core/vectors/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlArrayTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( core/vectors/tnlVectorTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlVectorTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} )         
            
endif()

ADD_TEST( tnl-unit-tests${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-unit-tests${mpiExt}${debugExt}  )                                                                       
                                                                       
