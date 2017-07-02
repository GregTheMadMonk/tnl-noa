SET( headers GridTypeResolver.h
             GridTypeResolver_impl.h
             MeshTypeResolver.h
             MeshTypeResolver_impl.h
             TypeResolver.h
             TypeResolver_impl.h
)

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/TypeResolver )
