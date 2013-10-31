ADD_SUBDIRECTORY( navier-stokes )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers/cfd )
   
set( tnl_implementation_solvers_cfd_SOURCES
     ${tnl_implementation_solvers_cfd_navier_stokes_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA)
   set( tnl_implementation_solvers_cfd_CUDA__SOURCES
        ${tnl_implementation_solvers_cfd_navier_stokes_CUDA__SOURCES}        
        PARENT_SCOPE )
endif()
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/cfd )