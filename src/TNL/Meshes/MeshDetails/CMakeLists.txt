ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( initializer )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )

SET( headers MeshEntityId.h
             MeshEntityIntegrityChecker.h
             MeshEntityOrientation.h
             MeshEntityReferenceOrientation.h
             MeshEntity_impl.h
             MeshIntegrityChecker.h
             MeshIntegrityCheckerLayer.h
             MeshWriterNetgen.h
             MeshWriterVTKLegacy.h
             Mesh_impl.h
             Traverser_impl.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails )
