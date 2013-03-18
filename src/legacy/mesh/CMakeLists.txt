ADD_SUBDIRECTORY( implementation )

SET( headers tnlDistributedGrid.h
             tnlGridOld.h )

set( tnl_legacy_mesh_SOURCES 
     ${tnl_legacy_mesh_implementation_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/legacy/mesh )