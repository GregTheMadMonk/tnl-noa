SET( headers EntityShape.h
             NetgenReader.h
             TNLReader.h
             VTKReader.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/Readers )
