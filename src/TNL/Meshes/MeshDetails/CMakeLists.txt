ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( initializer )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )

SET( headers MeshEntityId.h
             MeshEntityIntegrityChecker.h
             MeshEntityOrientation.h
             MeshEntity_impl.h
             MeshIntegrityChecker.h
             MeshIntegrityCheckerLayer.h
             MeshReaderNetgen.h
             MeshWriterNetgen.h
             MeshWriterVTKLegacy.h
             Mesh_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Meshes/MeshDetails )    
SET( tnl_mesh_SOURCES ${tnl_mesh_griddetails_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/MeshDetails )