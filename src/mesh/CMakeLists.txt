ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( grids )
ADD_SUBDIRECTORY( initializer )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )
ADD_SUBDIRECTORY( topologies )

SET( headers tnlGrid.h
             tnlDummyMesh.h
             tnlDimensionsTag.h
             tnlGnuplotWriter.h
             tnlMesh.h
             tnlMeshEntity.h
             tnlMeshEntityId.h
             tnlMeshReaderNetgen.h
             tnlMeshWriterNetgen.h
             tnlMeshIntegrityChecker.h
             tnlTraverser.h
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/mesh )    
SET( tnl_mesh_SOURCES
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )