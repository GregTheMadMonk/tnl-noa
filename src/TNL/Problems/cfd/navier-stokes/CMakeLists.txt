SET( headers NavierStokesSolver.h 
             NavierStokesSolver_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Solvers/cfd/navier-stokes )

if( BUILD_CUDA)
      set( tnl_solvers_cfd_navier_stokes_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_solvers_cfd_navier_stokes_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/solvers/cfd/navier-stokes )
