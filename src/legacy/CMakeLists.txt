ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( diff )
#ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( solvers )

SET( headers  )

set( tnl_legacy_SOURCES 
     ${tnl_legacy_mesh_SOURCES}
     ${tnl_legacy_solvers_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/legacy )