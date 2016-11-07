ADD_SUBDIRECTORY( TemplateExplicitInstantiation )

set( headers cuda-prefix-sum.h
             cuda-prefix-sum_impl.h
             cuda-reduction.h             
             cuda-reduction_impl.h
             reduction-operations.h
             CudaReduction.h
             CudaReduction_impl.h
             CudaReductionBuffer.h
             CublasWrapper.h
             CudaMultireductionKernel.h
             Multireduction.h
             Multireduction_impl.h
   )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Containers/Algorithms )
