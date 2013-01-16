ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( solvers )

SET( headers  )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation )

