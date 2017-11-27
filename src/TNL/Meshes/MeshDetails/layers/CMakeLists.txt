SET( headers MeshStorageLayer.h
             MeshSubentityStorageLayer.h
             MeshSuperentityStorageLayer.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/MeshDetails/layers )