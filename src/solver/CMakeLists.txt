SET( headers tnlBICGSolver.h
             tnlBICGStabSolver.h
             tnlCGSolver.h
             tnlEulerSolver.h
             tnlExplicitSolver.h
             tnlGMRESSolver.h
             tnlILUPreconditioner.h
             tnlMatrixSolver.h
             tnlMersonSolver.h
             tnlPETSCPreconditioner.h
             tnlPETSCSolver.h
             tnlPreconditioner.h
             tnlSORSolver.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solver )
