ADD_SUBDIRECTORY( cfd )
ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( ode )
ADD_SUBDIRECTORY( pde )
ADD_SUBDIRECTORY( preconditioners )

SET( headers tnlIterativeSolver.h
             tnlIterativeSolver_impl.h
             tnlBuildConfigTags.h
             tnlFastBuildConfigTag.h
             tnlMeshTypeResolver.h
             tnlMeshTypeResolver_impl.h
             tnlSolver.h
             tnlSolver_impl.h
             tnlSolverStarter.h
             tnlSolverStarter_impl.h
             tnlSolverInitiator.h
             tnlSolverInitiator_impl.h
             tnlSolverMonitor.h
             tnlSolverConfig.h
             tnlSolverConfig_impl.h
             tnlDummyProblem.h
             tnlIterativeSolverMonitor.h
             tnlIterativeSolverMonitor_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/solvers )
set( common_SOURCES ${CURRENT_DIR}/tnlIterativeSolver_impl.cpp )

if( BUILD_CUDA)
      set( tnl_solvers_CUDA__SOURCES
        ${tnl_solvers_linear_CUDA__SOURCES}
        ${tnl_solvers_cfd_CUDA__SOURCES}
        ${tnl_solvers_ode_CUDA__SOURCES}
        ${tnl_solvers_pde_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_solvers_SOURCES
     ${tnl_solvers_linear_SOURCES}
     ${tnl_solvers_cfd_SOURCES}
     ${tnl_solvers_ode_SOURCES}
     ${tnl_solvers_pde_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers )
