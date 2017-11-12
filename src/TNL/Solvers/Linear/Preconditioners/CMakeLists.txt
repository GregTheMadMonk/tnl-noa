SET( headers Dummy.h
             Diagonal.h
             Diagonal_impl.h
             ILU0.h
             ILU0_impl.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/Linear/Preconditioners )
