ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( pde )
ADD_SUBDIRECTORY( ode )

SET( headers tnlIterativeSolver_impl.h
             tnlIterativeSolverMonitor_impl.h
             tnlSolver_impl.h
             tnlSolverStarter_impl.h
             tnlSolverInitiator_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers ) 
set( tnl_implementation_solvers_SOURCES
     ${tnl_implementation_solvers_linear_SOURCES}
     ${tnl_implementation_solvers_ode_SOURCES}
     ${CURRENT_DIR}/tnlIterativeSolver_impl.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers )

