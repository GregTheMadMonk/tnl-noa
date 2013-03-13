ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( pde )
ADD_SUBDIRECTORY( ode )

SET( headers tnlIterativeSolverMonitor_impl.h
             tnlSolver_impl.h
             tnlSolverStarter_impl.h
             tnlSolverInitiator_impl.h )

set( tnl_implementation_solvers_SOURCES
     ${tnl_implementation_solvers_linear_SOURCES}
     ${tnl_implementation_solvers_ode_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers )

