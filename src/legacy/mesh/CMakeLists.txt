ADD_SUBDIRECTORY( implementation )

SET( headers tnlDistributedGrid.h
             tnlGridOld.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/legacy/mesh )