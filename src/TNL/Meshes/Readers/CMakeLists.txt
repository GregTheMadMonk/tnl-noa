SET( headers Netgen.h
             TNL.h
             VTKEntityType.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/Readers )
