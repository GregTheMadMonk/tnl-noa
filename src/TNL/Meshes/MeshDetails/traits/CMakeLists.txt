SET( headers MeshTraits.h
             MeshEntityTraits.h
             MeshSubentityTraits.h
             MeshSuperentityTraits.h
               )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/traits )