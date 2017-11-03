SET( headers BoundaryTagsLayer.h
             StorageLayer.h
             SubentityStorageLayer.h
             SuperentityStorageLayer.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/MeshDetails/MeshLayers )
