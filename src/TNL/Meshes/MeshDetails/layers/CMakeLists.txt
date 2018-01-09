SET( headers MeshStorageLayer.h
             MeshSubentityStorageLayer.h
             MeshSuperentityStorageLayer.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/layers )