SET( headers tnlDimensionsTrait.h
             tnlMeshTrait.h
             tnlStorageTrait.h
               )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/traits )