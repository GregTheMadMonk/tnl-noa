ADD_SUBDIRECTORY( implementation )

SET( headers tnlGMRESSolver.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear/krylov )
