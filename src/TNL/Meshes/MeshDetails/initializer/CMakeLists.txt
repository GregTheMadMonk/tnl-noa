SET( headers MeshEntityInitializer.h
             MeshEntitySeed.h 
             MeshEntitySeedKey.h
             MeshInitializer.h
             MeshSubentitySeedCreator.h
             MeshSuperentityStorageInitializer.h
 )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Meshes/MeshDetails/initializer )    
SET( tnl_mesh_initializer_SOURCES
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/initializer )