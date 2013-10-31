set( headers tnlNavierStokesSolver_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers/cfd/navier-stokes )

if( BUILD_CUDA)
      set( tnl_implementation_solvers_cfd_navier_stokes_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_solvers_cfd_navier_stokes_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/cfd/navier-stokes )
