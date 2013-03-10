ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( ode )
ADD_SUBDIRECTORY( pde )
ADD_SUBDIRECTORY( preconditioners )

SET( headers tnlIterativeSolver.h
             tnlSolver.h
             tnlSolverStarter.h
             tnlSolverInitiator.h
             tnlSolverMonitor.h
             tnlSimpleIterativeSolverMonitor.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers )
