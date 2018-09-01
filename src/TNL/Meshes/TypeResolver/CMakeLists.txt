SET( headers GridTypeResolver.h
             GridTypeResolver_impl.h
             MeshTypeResolver.h
             MeshTypeResolver_impl.h
             TypeResolver.h
             TypeResolver_impl.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/TypeResolver )
