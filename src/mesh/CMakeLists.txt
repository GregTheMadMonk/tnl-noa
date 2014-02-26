ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )
ADD_SUBDIRECTORY( topologies )

SET( headers tnlGrid.h
             tnlIdenticalGridGeometry.h
             tnlLinearGridGeometry.h
             tnlDummyMesh.h
             tnlGnuplotWriter.h
             tnlMesh.h
             tnlMeshEntity.h
             tnlMeshEntityKey.h
             tnlMeshReaderNetgen.h
             tnlMeshWriterNetgen.h
             tnlMeshInitializer.h
             tnlMeshEntityInitializer.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )