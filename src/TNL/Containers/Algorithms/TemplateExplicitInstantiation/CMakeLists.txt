if( ${WITH_TEMPLATES_INSTANTIATION} )

   SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Containers/Algorithms/TemplateExplicitInstantiation )
   set( common_SOURCES
        ${CURRENT_DIR}/VectorOperationsHost_impl.cpp
   )
   IF( BUILD_CUDA )
      set( tnl_core_cuda_CUDA__SOURCES
           ${common_SOURCES}
           ${CURRENT_DIR}/ArrayOperationsHost_impl.cu
           ${CURRENT_DIR}/ArrayOperationsCuda_impl.cu
           ${CURRENT_DIR}/cuda-reduction-sum_impl.cu
           ${CURRENT_DIR}/cuda-reduction-min_impl.cu
           ${CURRENT_DIR}/cuda-reduction-max_impl.cu
           ${CURRENT_DIR}/cuda-reduction-abs-sum_impl.cu
           ${CURRENT_DIR}/cuda-reduction-abs-min_impl.cu
           ${CURRENT_DIR}/cuda-reduction-abs-max_impl.cu
           ${CURRENT_DIR}/cuda-reduction-and_impl.cu
           ${CURRENT_DIR}/cuda-reduction-or_impl.cu
           ${CURRENT_DIR}/cuda-reduction-l2-norm_impl.cu
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
           ${CURRENT_DIR}/cuda-reduction-diff-l2-norm_impl.cu
           ${CURRENT_DIR}/cuda-reduction-diff-lp-norm_impl.cu
           ${CURRENT_DIR}/cuda-prefix-sum_impl.cu
           ${CURRENT_DIR}/VectorOperationsCuda_impl.cu
           PARENT_SCOPE )
   ELSE()
      set( common_SOURCES
           ${common_SOURCES}
           ${CURRENT_DIR}/ArrayOperationsHost_impl.cpp
           ${CURRENT_DIR}/ArrayOperationsCuda_impl.cpp
      )
   ENDIF()

   set( tnl_core_cuda_SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/cuda-reduction_impl.cpp
        PARENT_SCOPE )
endif()
