SET( headers container.h
             id.h
             mesh_entity.h
             mesh_entity_border.h
             mesh_entity_coborder.h
             mesh_entity_key.h
             mesh_initializer.h
             mesh_layer.h
             mesh_pointer.h
             mesh_tags.h
             point.h
             range.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/detail )