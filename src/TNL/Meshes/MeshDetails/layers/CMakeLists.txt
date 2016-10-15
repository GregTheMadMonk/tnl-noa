SET( headers MeshStorageLayer.h
             MeshSubentityStorageLayer.h
             MeshSuperentityAccess.h
             MeshSuperentityAccessor.h
             MeshSuperentityStorageLayer.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/layers )
