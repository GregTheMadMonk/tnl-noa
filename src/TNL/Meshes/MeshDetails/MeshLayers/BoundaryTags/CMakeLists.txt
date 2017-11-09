SET( headers ConfigValidator.h
             Layer.h
             LayerFamily.h
             Initializer.h
             Traits.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/MeshDetails/MeshLayers/BoundaryTags )
