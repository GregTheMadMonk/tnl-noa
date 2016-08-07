SET( headers tnlDummyPreconditioner.h
             tnlDiagonalPreconditioner.h
             tnlDiagonalPreconditioner_impl.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/Linear/preconditioners )
