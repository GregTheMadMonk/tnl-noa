ADD_SUBDIRECTORY( TemplateExplicitInstantiation )

set( headers cuda-prefix-sum.h
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
   )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Containers/Algorithms )
