SET( headers AsymptoteWriter.h
             NetgenWriter.h
             VTKWriter.h
             VTKWriter_impl.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/Writers )
