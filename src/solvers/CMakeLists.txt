ADD_SUBDIRECTORY( cfd )
ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( ode )
ADD_SUBDIRECTORY( pde )
ADD_SUBDIRECTORY( preconditioners )

SET( headers tnlIterativeSolver.h
             tnlConfigTags.h
             tnlFastBuildConfig.h
             tnlMeshTypeResolver.h
             tnlSolver.h
             tnlSolverStarter.h
             tnlSolverInitiator.h
             tnlSolverMonitor.h
             tnlSolverConfig.h
             tnlDummyProblem.h
             tnlIterativeSolverMonitor.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers )
