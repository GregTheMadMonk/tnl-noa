ADD_SUBDIRECTORY( pde )

SET( headers  )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers )

