set( headers godunov.h
	     godunov1D_impl.h
             godunov2D_impl.h
             godunov3D_impl.h)

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/operators/godunov )

if( BUILD_CUDA)
      set( tnl_implementation_operators_godunov_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_operators_godunov_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Operators/Godunov/ )
