ADD_SUBDIRECTORY( BoundaryTags )

SET( headers StorageLayer.h
             SubentityStorageLayer.h
             SuperentityStorageLayer.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/MeshDetails/MeshLayers )
