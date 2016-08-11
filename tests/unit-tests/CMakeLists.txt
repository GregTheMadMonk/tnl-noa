ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( matrices )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( functions )
ADD_SUBDIRECTORY( operators )
ADD_SUBDIRECTORY( solver )

ADD_TEST( core/tnlStringTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlStringTest${mpiExt}${debugExt} )
ADD_TEST( core/tnlListTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlListTest${mpiExt}${debugExt} )
ADD_TEST( core/tnlObjectTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlObjectTest${mpiExt}${debugExt} )
ADD_TEST( core/arrays/tnlArrayOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayOperationsTest${mpiExt}${debugExt} )
ADD_TEST( core/arrays/tnlArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest${mpiExt}${debugExt} )
ADD_TEST( core/arrays/tnlStaticArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlStaticArrayTest${mpiExt}${debugExt} )
ADD_TEST( core/arrays/tnlMultiArrayTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultiArrayTest${mpiExt}${debugExt} )
ADD_TEST( core/vectors/tnlVectorOperationsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest${mpiExt}${debugExt} )
ADD_TEST( core/vectors/tnlVectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorTest${mpiExt}${debugExt} )
ADD_TEST( core/vectors/tnlStaticVectorTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlStaticVectorTest${mpiExt}${debugExt} )

ADD_TEST( core/multimaps/tnlEllpackIndexMultimapTest{mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackIndexMultimapTest${mpiExt}${debugExt} )

ADD_TEST( matrices/tnlDenseTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlTridiagonalTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlMultidiagonalTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlEllpackTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlSlicedEllpackTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlChunkedEllpackTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlCSRTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRTest${mpiExt}${debugExt} )

ADD_TEST( mesh/tnlGridTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlGridTest${mpiExt}${debugExt} )
ADD_TEST( mesh/MeshEntityTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${mpiExt}${debugExt} )
ADD_TEST( mesh/MeshTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshTest${mpiExt}${debugExt} )

ADD_TEST( functions/OperatorFunctionTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/OperatorFunctionTest${mpiExt}${debugExt} )

ADD_TEST( operators/diffusion/LinearDiffusionTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/LinearDiffusionTest${mpiExt}${debugExt} )
ADD_TEST( operators/diffusion/FiniteDifferencesTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/FiniteDifferencesTest${mpiExt}${debugExt} )

if( BUILD_CUDA )
   ADD_TEST( core/cuda/tnlCudaTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCudaTest${mpiExt}${debugExt} )
   
   ADD_TEST( core/cuda/tnl-reduction-test${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-reduction-test${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/cuda/tnl-reduction-test${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlCudaTest${mpiExt}${debugExt} )   
   
   ADD_TEST( core/arrays/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnl-reduction-test${mpiExt}${debugExt} )
   
   ADD_TEST( core/arrays/tnlArrayTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlArrayTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlArrayOperationsTest-cuda${mpiExt}${debugExt} )
         
   ADD_TEST( core/arrays/tnlMultiArrayTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultiArrayTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlMultiArrayTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlArrayTest-cuda${mpiExt}${debugExt} )        
   
   ADD_TEST( core/vectors/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlMultiArrayTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( core/vectors/tnlVectorTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlVectorTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlVectorOperationsTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlDenseTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlDenseTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlVectorTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlTridiagonalTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlTridiagonalTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlDenseTest-cuda${mpiExt}${debugExt} )         
            
   ADD_TEST( matrices/tnlMultidiagonalTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlMultidiagonalTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlTridiagonalTest-cuda${mpiExt}${debugExt} )         
            
   ADD_TEST( matrices/tnlEllpackTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlEllpackTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlMultidiagonalTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlSlicedEllpackTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlSlicedEllpackTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlEllpackTest-cuda${mpiExt}${debugExt} )         
   
   ADD_TEST( matrices/tnlChunkedEllpackTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlChunkedEllpackTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlSlicedEllpackTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlCSRTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlCSRTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlChunkedEllpackTest-cuda${mpiExt}${debugExt} )                     
                                    
endif()
                                                                       
                                                                       
