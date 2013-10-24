set( headers tnlNavierStokes_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/schemes/navier-stokes )

if( BUILD_CUDA)
      set( tnl_implementation_schemes_navier_stokes_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_schemes_navier_stokes_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/schemes/navier-stokes )
