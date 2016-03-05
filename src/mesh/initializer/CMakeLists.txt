SET( headers tnlMeshEntityInitializer.h
             tnlMeshEntitySeed.h 
             tnlMeshEntitySeedKey.h
             tnlMeshInitializer.h
             tnlMeshSubentitySeedCreator.h
             tnlMeshSuperentityStorageInitializer.h
 )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/mesh/initializer )    
SET( tnl_mesh_initializer_SOURCES
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/initializer )