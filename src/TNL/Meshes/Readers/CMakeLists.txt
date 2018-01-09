SET( headers EntityShape.h
             NetgenReader.h
             TNLReader.h
             VTKReader.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/Readers )
