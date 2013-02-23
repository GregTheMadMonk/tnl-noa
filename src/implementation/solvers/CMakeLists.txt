ADD_SUBDIRECTORY( linear )
ADD_SUBDIRECTORY( pde )

SET( headers tnlProblemSolver_impl.h )

set( tnl_implementation_solvers_SOURCES
     ${tnl_implementation_solvers_linear_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers )

