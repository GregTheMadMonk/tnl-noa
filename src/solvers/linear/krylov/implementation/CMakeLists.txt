SET( headers tnlCGSolver_impl.h
             tnlBICGStabSolver_impl.h
             tnlGMRESSolver_impl.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear/krylov/implementation )
