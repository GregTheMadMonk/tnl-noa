ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )
ADD_SUBDIRECTORY( topologies )

SET( headers tnlGrid.h
             tnlGrid1D.h
             tnlGrid1D_impl.h
             tnlGrid2D.h
             tnlGrid2D_impl.h
             tnlGrid3D.h
             tnlGrid3D_impl.h
             tnlDummyMesh.h
             tnlDimensionsTag.h
             tnlGnuplotWriter.h
             tnlMesh.h
             tnlMeshEntity.h
             tnlMeshEntityId.h
             tnlMeshEntitySeedKey.h
             tnlMeshReaderNetgen.h
             tnlMeshWriterNetgen.h
             tnlMeshInitializer.h
             tnlMeshIntegrityChecker.h
             tnlMeshEntityInitializer.h
             tnlMeshSuperentityStorageInitializer.h
             tnlTraverser.h
             tnlTraverser_Grid1D.h
             tnlTraverser_Grid1D_impl.h 
             tnlTraverser_Grid2D.h
             tnlTraverser_Grid2D_impl.h 
             tnlTraverser_Grid3D.h
             tnlTraverser_Grid3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/mesh )    
SET( tnl_mesh_SOURCES
     ${CURRENT_DIR}/tnlGrid_impl.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )