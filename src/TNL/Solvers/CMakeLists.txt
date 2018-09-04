ADD_SUBDIRECTORY( Linear )
ADD_SUBDIRECTORY( ODE )
ADD_SUBDIRECTORY( PDE )

SET( headers IterativeSolver.h
             IterativeSolver_impl.h
             BuildConfigTags.h
             FastBuildConfigTag.h
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

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Solvers )
