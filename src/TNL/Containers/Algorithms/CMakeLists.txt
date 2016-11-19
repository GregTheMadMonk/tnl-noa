ADD_SUBDIRECTORY( TemplateExplicitInstantiation )

set( headers ArrayOperations.h
             ArrayOperationsHost_impl.h
             ArrayOperationsCuda_impl.h
             cuda-prefix-sum.h
             cuda-prefix-sum_impl.h
             reduction-operations.h
             CublasWrapper.h
             CudaMultireductionKernel.h
             CudaReductionBuffer.h
             CudaReductionKernel.h
             Multireduction.h
             Multireduction_impl.h
             Reduction.h
             Reduction_impl.h
             VectorOperations.h
             VectorOperationsHost_impl.h
             VectorOperationsCuda_impl.h
   )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Containers/Algorithms )
