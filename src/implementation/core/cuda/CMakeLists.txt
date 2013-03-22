SET( headers reduction_impl.h
             reduction-operations_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core/cuda )

IF( BUILD_CUDA )
   set( tnl_implementation_core_cuda_CUDA__SOURCES
        ${CURRENT_DIR}/reduction-operations_impl.cu
        PARENT_SCOPE )
endif()        

set( tnl_implementation_core_cuda_SOURCES 
     ${CURRENT_DIR}/device-check.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation )

