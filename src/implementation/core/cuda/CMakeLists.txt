SET( headers reduction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core/cuda )

set( tnl_implementation_core_cuda_SOURCES 
     ${CURRENT_DIR}/device-check.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation )

