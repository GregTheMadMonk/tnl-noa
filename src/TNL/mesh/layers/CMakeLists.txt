SET( headers tnlMeshStorageLayer.h
             tnlMeshSubentityStorageLayer.h
             tnlMeshSuperentityStorageLayer.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/layers )