SET( headers tnlDimensionsTraits.h
             tnlMeshTraits.h
             tnlStorageTraits.h
             tnlMeshSubentitiesTraits.h
               )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/traits )