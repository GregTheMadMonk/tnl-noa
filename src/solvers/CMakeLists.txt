ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( ode )
ADD_SUBDIRECTORY( preconditioners )

SET( headers tnlIterativeSolver.h
             tnlSolverMonitor.h
             tnlSimpleIterativeSolverMonitor.h
             tnlBICGSolver.h
             tnlBICGStabSolver.h
             tnlCGSolver.h
             tnlGMRESSolverOld.h
             tnlILUPreconditioner.h
             tnlMatrixSolver.h             
             tnlPETSCPreconditioner.h
             tnlPETSCSolver.h
             tnlPreconditioner.h
             tnlSORSolver.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers )
