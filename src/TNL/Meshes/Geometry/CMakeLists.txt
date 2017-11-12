SET( headers getEntityCenter.h
             getEntityMeasure.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/Geometry )
