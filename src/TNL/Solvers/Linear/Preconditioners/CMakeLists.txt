SET( headers Dummy.h
             Diagonal.h
             Diagonal_impl.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/Linear/Preconditioners )
