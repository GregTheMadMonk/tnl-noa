ADD_SUBDIRECTORY( BoundaryTags )

SET( headers StorageLayer.h
             SubentityStorageLayer.h
             SuperentityStorageLayer.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/MeshLayers )
