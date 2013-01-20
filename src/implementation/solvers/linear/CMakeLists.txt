ADD_SUBDIRECTORY( krylov )
ADD_SUBDIRECTORY( stationary )

SET( headers tnlLinearResidueGetter_impl.h
   )
   
set( tnl_implementation_solvers_linear_SOURCES
     ${tnl_implementation_solvers_linear_krylov_SOURCES}
     ${tnl_implementation_solvers_linear_stationary_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/linear )