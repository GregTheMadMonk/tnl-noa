SET( headers NetgenReader.h
             TNLReader.h
             VTKEntityType.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/Readers )
