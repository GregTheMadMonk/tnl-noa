ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( matrices )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( functions )
ADD_SUBDIRECTORY( operators )
ADD_SUBDIRECTORY( solver )


ADD_TEST( core/arrays/tnlArrayOperationsTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayOperationsTest${mpiExt} )
ADD_TEST( core/arrays/tnlArrayTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest${mpiExt} )
ADD_TEST( core/arrays/tnlStaticArrayTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlStaticArrayTest${mpiExt} )
ADD_TEST( core/arrays/tnlMultiArrayTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultiArrayTest${mpiExt} )
ADD_TEST( core/vectors/tnlVectorOperationsTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest${mpiExt} )
ADD_TEST( core/vectors/tnlVectorTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorTest${mpiExt} )
ADD_TEST( core/vectors/tnlStaticVectorTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlStaticVectorTest${mpiExt} )

ADD_TEST( core/multimaps/tnlEllpackIndexMultimapTest{mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackIndexMultimapTest${mpiExt} )

ADD_TEST( matrices/tnlDenseTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseTest${mpiExt} )
ADD_TEST( matrices/tnlTridiagonalTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalTest${mpiExt} )
ADD_TEST( matrices/tnlMultidiagonalTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalTest${mpiExt} )
ADD_TEST( matrices/tnlEllpackTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackTest${mpiExt} )
ADD_TEST( matrices/tnlSlicedEllpackTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackTest${mpiExt} )
ADD_TEST( matrices/tnlChunkedEllpackTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackTest${mpiExt} )
ADD_TEST( matrices/tnlCSRTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRTest${mpiExt} )

ADD_TEST( mesh/tnlGridTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlGridTest${mpiExt} )
ADD_TEST( mesh/MeshEntityTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${mpiExt} )
ADD_TEST( mesh/MeshTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/MeshTest${mpiExt} )

ADD_TEST( functions/OperatorFunctionTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/OperatorFunctionTest${mpiExt} )

ADD_TEST( operators/diffusion/LinearDiffusionTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/LinearDiffusionTest${mpiExt} )
ADD_TEST( operators/diffusion/FiniteDifferencesTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/FiniteDifferencesTest${mpiExt} )

if( BUILD_CUDA )
   ADD_TEST( core/cuda/tnlCudaTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCudaTest${mpiExt} )
   
   ADD_TEST( core/cuda/tnl-reduction-test${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-reduction-test${mpiExt} )
   SET_TESTS_PROPERTIES ( core/cuda/tnl-reduction-test${mpiExt} PROPERTIES DEPENDS core/cuda/tnlCudaTest${mpiExt} )   
   
   ADD_TEST( core/arrays/tnlArrayOperationsTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayOperationsTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlArrayOperationsTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnl-reduction-test${mpiExt} )
   
   ADD_TEST( core/arrays/tnlArrayTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlArrayTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlArrayTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlArrayOperationsTest-cuda${mpiExt} )
         
   ADD_TEST( core/arrays/tnlMultiArrayTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultiArrayTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( core/arrays/tnlMultiArrayTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlArrayTest-cuda${mpiExt} )        
   
   ADD_TEST( core/vectors/tnlVectorOperationsTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorOperationsTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlVectorOperationsTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlMultiArrayTest-cuda${mpiExt} )
   
   ADD_TEST( core/vectors/tnlVectorTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlVectorTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( core/vectors/tnlVectorTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlVectorOperationsTest-cuda${mpiExt} )
   
   ADD_TEST( matrices/tnlDenseTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlDenseTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlVectorTest-cuda${mpiExt} )
   
   ADD_TEST( matrices/tnlTridiagonalTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlTridiagonalTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlDenseTest-cuda${mpiExt} )         
            
   ADD_TEST( matrices/tnlMultidiagonalTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlMultidiagonalTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlTridiagonalTest-cuda${mpiExt} )         
            
   ADD_TEST( matrices/tnlEllpackTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlEllpackTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlMultidiagonalTest-cuda${mpiExt} )
   
   ADD_TEST( matrices/tnlSlicedEllpackTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlSlicedEllpackTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlEllpackTest-cuda${mpiExt} )         
   
   ADD_TEST( matrices/tnlChunkedEllpackTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlChunkedEllpackTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlSlicedEllpackTest-cuda${mpiExt} )
   
   ADD_TEST( matrices/tnlCSRTest-cuda${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRTest-cuda${mpiExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlCSRTest-cuda${mpiExt} PROPERTIES DEPENDS core/cuda/tnlChunkedEllpackTest-cuda${mpiExt} )                     
                                    
endif()
                                                                       
                                                                       
