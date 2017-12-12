SET( headers EntityInitializer.h
             EntitySeed.h 
             Initializer.h
             SubentitySeedsCreator.h
             SuperentityStorageInitializer.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/MeshDetails/initializer )
