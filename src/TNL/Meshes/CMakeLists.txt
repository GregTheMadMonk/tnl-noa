ADD_SUBDIRECTORY( GridDetails )
ADD_SUBDIRECTORY( MeshDetails )
ADD_SUBDIRECTORY( DistributedMeshes )
ADD_SUBDIRECTORY( Topologies )

SET( headers Grid.h
             GridEntity.h
             GridEntityConfig.h
             DummyMesh.h
             MeshDimensionTag.h             
             Mesh.h
             MeshEntity.h
             Traverser.h
             MeshConfigBase.h
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Meshes )    
SET( tnl_mesh_SOURCES ${tnl_mesh_griddetails_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes )
