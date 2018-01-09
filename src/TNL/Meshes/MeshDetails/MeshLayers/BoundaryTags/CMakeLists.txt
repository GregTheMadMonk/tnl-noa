SET( headers ConfigValidator.h
             Layer.h
             LayerFamily.h
             Initializer.h
             Traits.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/MeshLayers/BoundaryTags )
