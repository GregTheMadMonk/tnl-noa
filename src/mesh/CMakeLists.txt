ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )
ADD_SUBDIRECTORY( topologies )

SET( headers tnlGrid.h
             tnlIdenticalGridGeometry.h
             tnlLinearGridGeometry.h
             tnlDummyMesh.h
             tnlGnuplotWriter.h
             tnlMeshStorageLayer.h  )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )