ADD_SUBDIRECTORY( Linear )
ADD_SUBDIRECTORY( ODE )
ADD_SUBDIRECTORY( PDE )


SET( headers IterativeSolver.h
             IterativeSolver_impl.h
             BuildConfigTags.h
             FastBuildConfigTag.h
             MeshTypeResolver.h
             MeshTypeResolver_impl.h
             Solver.h
             Solver_impl.h
             SolverStarter.h
             SolverStarter_impl.h
             SolverInitiator.h
             SolverInitiator_impl.h
             SolverMonitor.h
             SolverConfig.h
             SolverConfig_impl.h
             DummyProblem.h
             IterativeSolverMonitor.h
             IterativeSolverMonitor_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Solvers )
set( common_SOURCES ${CURRENT_DIR}/IterativeSolver_impl.cpp )

if( BUILD_CUDA)
      set( tnl_solvers_CUDA__SOURCES
        ${tnl_solvers_linear_CUDA__SOURCES}
        ${tnl_solvers_ode_CUDA__SOURCES}
        ${tnl_solvers_pde_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_solvers_SOURCES
     ${tnl_solvers_linear_SOURCES}
     ${tnl_solvers_ode_SOURCES}
     ${tnl_solvers_pde_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Solvers )
