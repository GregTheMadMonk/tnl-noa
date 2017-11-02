ADD_SUBDIRECTORY( initializer )
ADD_SUBDIRECTORY( layers )
ADD_SUBDIRECTORY( traits )

SET( headers ConfigValidator.h
             IndexPermutationApplier.h
             MeshEntityIndex.h
             MeshEntityIntegrityChecker.h
             MeshEntityOrientation.h
             MeshEntityReferenceOrientation.h
             MeshEntity_impl.h
             MeshIntegrityChecker.h
             MeshIntegrityCheckerLayer.h
             Mesh_impl.h
             Traverser_impl.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails )
