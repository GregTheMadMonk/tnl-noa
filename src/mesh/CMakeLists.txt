ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )
ADD_SUBDIRECTORY( topologies )

SET( headers tnlGrid.h
             tnlDummyMesh.h
             tnlGnuplotWriter.h
             tnlMesh.h
             tnlMeshEntity.h
             tnlMeshEntityId.h
             tnlMeshEntityKey.h
             tnlMeshReaderNetgen.h
             tnlMeshWriterNetgen.h
             tnlMeshInitializer.h
             tnlMeshEntityInitializer.h
             tnlMeshSuperentityInitializerLayer.h
             tnlTraversal.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )