SET( headers tnlDimensionsTraits.h
             tnlMeshTraits.h
             tnlStorageTraits.h
             tnlMeshEntitiesTraits.h
             tnlMeshEntitiesTag.h
             tnlMeshSubentitiesTraits.h
             tnlMeshSuperentitiesTraits.h
               )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/traits )