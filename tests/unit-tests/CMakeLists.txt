ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( matrices )
ADD_SUBDIRECTORY( mesh )
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

ADD_TEST( matrices/tnlDenseMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlTridiagonalMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlMultidiagonalMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlEllpackMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlSlicedEllpackMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlChunkedEllpackMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackMatrixTest${mpiExt}${debugExt} )
ADD_TEST( matrices/tnlCSRMatrixTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRMatrixTest${mpiExt}${debugExt} )

ADD_TEST( mesh/tnlGridTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlGridTest${mpiExt}${debugExt} )
ADD_TEST( mesh/tnlMeshEntityTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMeshEntityTest${mpiExt}${debugExt} )
ADD_TEST( mesh/tnlMeshTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMeshTest${mpiExt}${debugExt} )

ADD_TEST( operators/diffusion/tnlLinearDiffusionTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlLinearDiffusionTest${mpiExt}${debugExt} )

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
   
   ADD_TEST( matrices/tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlDenseMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlDenseMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlVectorTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlDenseMatrixTest-cuda${mpiExt}${debugExt} )         
            
   ADD_TEST( matrices/tnlMultidiagonalMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlMultidiagonalMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlMultidiagonalMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} )         
            
   ADD_TEST( matrices/tnlEllpackMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlEllpackMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlEllpackMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlMultidiagonalMatrixTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlSlicedEllpackMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSlicedEllpackMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlSlicedEllpackMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlEllpackMatrixTest-cuda${mpiExt}${debugExt} )         
   
   ADD_TEST( matrices/tnlChunkedEllpackMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlChunkedEllpackMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlChunkedEllpackMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlSlicedEllpackMatrixTest-cuda${mpiExt}${debugExt} )
   
   ADD_TEST( matrices/tnlCSRMatrixTest-cuda${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlCSRMatrixTest-cuda${mpiExt}${debugExt} )
   SET_TESTS_PROPERTIES ( matrices/tnlCSRMatrixTest-cuda${mpiExt}${debugExt} PROPERTIES DEPENDS core/cuda/tnlChunkedEllpackMatrixTest-cuda${mpiExt}${debugExt} )                     
                                    
endif()
                                                                       
                                                                       
