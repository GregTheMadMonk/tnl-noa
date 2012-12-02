ADD_SUBDIRECTORY( implementation )

SET( headers tnlCGSolver.h
             tnlBICGStabSolver.h
             tnlGMRESSolver.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear/krylov )
