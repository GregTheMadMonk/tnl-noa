set( headers upwind1D_impl.h
             upwind2D_impl.h
             upwind3D_impl.h)

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/operators/upwind )

if( BUILD_CUDA)
      set( tnl_implementation_operators_upwind_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_operators_upwind_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/operators/upwind )
