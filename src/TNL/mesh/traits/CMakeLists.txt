SET( headers tnlMeshTraits.h
             tnlMeshEntityTraits.h
             tnlMeshSubentityTraits.h
             tnlMeshSuperentityTraits.h
               )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/traits )