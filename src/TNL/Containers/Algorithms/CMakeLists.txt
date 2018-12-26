set( headers ArrayIO.h
             ArrayOperations.h
             ArrayOperationsHost_impl.h
             ArrayOperationsCuda_impl.h
             ArrayOperationsMIC_impl.h
             cuda-prefix-sum.h
             cuda-prefix-sum_impl.h
             CudaMultireductionKernel.h
             CudaReductionBuffer.h
             CudaReductionKernel.h
             Multireduction.h
             Multireduction_impl.h
             Reduction.h
             Reduction_impl.h
             ReductionOperations.h
             VectorOperations.h
             VectorOperationsHost_impl.h
             VectorOperationsCuda_impl.h
             VectorOperationsMIC_impl.h
   )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Containers/Algorithms )
