ADD_SUBDIRECTORY( krylov )
ADD_SUBDIRECTORY( stationary )

SET( headers tnlLinearResidueGetter.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear )