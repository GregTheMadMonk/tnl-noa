SET( headers MeshTraits.h
             MeshEntityTraits.h
             MeshSubentityTraits.h
             MeshSuperentityTraits.h
               )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/MeshDetails/traits )