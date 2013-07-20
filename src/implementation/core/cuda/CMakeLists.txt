SET( headers cuda-reduction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core/cuda )
#set( common_SOURCES  ) 
IF( BUILD_CUDA )
   set( tnl_implementation_core_cuda_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/cuda-reduction_impl.cu
        PARENT_SCOPE )        
endif() 

set( tnl_implementation_core_cuda_SOURCES
     ${common_SOURCES}
     ${CURRENT_DIR}/cuda-reduction_impl.cpp     
     PARENT_SCOPE )               
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/core/cuda )

