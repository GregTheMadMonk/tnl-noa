SET( headers MeshEntityStorageRebinder.h
             MeshStorageLayer.h
             MeshSubentityAccess.h
             MeshSubentityStorageLayer.h
             MeshSuperentityAccess.h
             MeshSuperentityStorageLayer.h
             MeshBoundaryTagsLayer.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/layers )
