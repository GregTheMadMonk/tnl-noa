SET( headers AsymptoteWriter.h
             NetgenWriter.h
             VTKWriter.h
             VTKWriter_impl.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/Writers )
