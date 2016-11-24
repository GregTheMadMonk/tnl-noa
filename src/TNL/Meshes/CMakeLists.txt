ADD_SUBDIRECTORY( GridDetails )
ADD_SUBDIRECTORY( MeshDetails )
ADD_SUBDIRECTORY( Readers )
ADD_SUBDIRECTORY( Topologies )
ADD_SUBDIRECTORY( TypeResolver )

SET( headers Grid.h
             GridEntity.h
             GridEntityConfig.h
             DimensionTag.h
             DummyMesh.h
             Mesh.h
             MeshEntity.h
             Traverser.h
             MeshConfigBase.h
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Meshes )    
SET( tnl_mesh_SOURCES ${tnl_mesh_griddetails_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes )
