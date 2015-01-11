set( headers cuda-prefix-sum.h
             cuda-prefix-sum_impl.h
             cuda-reduction.h             
             cuda-reduction_impl.h
             reduction-operations.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core/cuda ) 
IF( BUILD_CUDA )
   set( tnl_core_cuda_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/cuda-reduction-sum_impl.cu
        ${CURRENT_DIR}/cuda-reduction-min_impl.cu
        ${CURRENT_DIR}/cuda-reduction-max_impl.cu
        ${CURRENT_DIR}/cuda-reduction-abs-sum_impl.cu
        ${CURRENT_DIR}/cuda-reduction-abs-min_impl.cu
        ${CURRENT_DIR}/cuda-reduction-abs-max_impl.cu
        ${CURRENT_DIR}/cuda-reduction-and_impl.cu
        ${CURRENT_DIR}/cuda-reduction-or_impl.cu
        ${CURRENT_DIR}/cuda-reduction-lp-norm_impl.cu
        ${CURRENT_DIR}/cuda-reduction-equalities_impl.cu
        ${CURRENT_DIR}/cuda-reduction-inequalities_impl.cu
        ${CURRENT_DIR}/cuda-reduction-scalar-product_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-sum_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-min_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-max_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-abs-sum_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-abs-min_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-abs-max_impl.cu
        ${CURRENT_DIR}/cuda-reduction-diff-lp-norm_impl.cu        
        ${CURRENT_DIR}/cuda-prefix-sum_impl.cu
        PARENT_SCOPE )        
endif() 

set( tnl_core_cuda_SOURCES
     ${common_SOURCES}
     ${CURRENT_DIR}/cuda-reduction_impl.cpp
     ${CURRENT_DIR}/cuda-prefix-sum_impl.cpp     
     PARENT_SCOPE )               
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/cuda )