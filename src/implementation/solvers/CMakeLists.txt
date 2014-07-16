ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( cfd )
ADD_SUBDIRECTORY( pde )
ADD_SUBDIRECTORY( ode )

SET( headers tnlIterativeSolver_impl.h
             tnlIterativeSolverMonitor_impl.h
             tnlMeshTypeResolver_impl.h
             tnlSolver_impl.h
             tnlSolverConfig_impl.h
             tnlSolverStarter_impl.h
             tnlSolverInitiator_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers )
set( common_SOURCES ${CURRENT_DIR}/tnlIterativeSolver_impl.cpp )

if( BUILD_CUDA)
      set( tnl_implementation_solvers_CUDA__SOURCES
        ${tnl_implementation_solvers_linear_CUDA__SOURCES}
        ${tnl_implementation_solvers_cfd_CUDA__SOURCES}
        ${tnl_implementation_solvers_ode_CUDA__SOURCES}
        ${tnl_implementation_solvers_pde_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_solvers_SOURCES
     ${tnl_implementation_solvers_linear_SOURCES}
     ${tnl_implementation_solvers_cfd_SOURCES}
     ${tnl_implementation_solvers_ode_SOURCES}
     ${tnl_implementation_solvers_pde_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers )

