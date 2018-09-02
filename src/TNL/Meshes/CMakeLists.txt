ADD_SUBDIRECTORY( Geometry )
ADD_SUBDIRECTORY( GridDetails )
ADD_SUBDIRECTORY( MeshDetails )
ADD_SUBDIRECTORY( DistributedMeshes )
ADD_SUBDIRECTORY( Readers )
ADD_SUBDIRECTORY( Topologies )
ADD_SUBDIRECTORY( TypeResolver )
ADD_SUBDIRECTORY( Writers )

SET( headers BuildConfigTags.h
             DefaultConfig.h
             DimensionTag.h
             DummyMesh.h
             GridEntityConfig.h
             GridEntity.h
             Grid.h
             MeshBuilder.h
             MeshEntity.h
             Mesh.h
             Traverser.h
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Meshes )    
SET( tnl_mesh_SOURCES ${tnl_mesh_griddetails_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes )

