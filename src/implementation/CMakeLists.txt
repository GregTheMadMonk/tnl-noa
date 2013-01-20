ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( solvers )

SET( headers  )

set( tnl_implementation_SOURCES 
     ${tnl_implementation_core_SOURCES}
     ${tnl_implementation_solvers_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation )

