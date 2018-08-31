set( tnl_fast_sweeping_map_SOURCES
#     MainBuildConfig.h
#     tnlFastSweepingMap2D_impl.h
#     tnlFastSweepingMap.h
#     fastSweepingMapConfig.h 
     main.cpp)


IF(  BUILD_CUDA ) 
	CUDA_ADD_EXECUTABLE(fast-sweeping-map main.cu)
ELSE(  BUILD_CUDA )                
	ADD_EXECUTABLE(fast-sweeping-map main.cpp)
ENDIF( BUILD_CUDA )
target_link_libraries (fast-sweeping-map tnl )


INSTALL( TARGETS fast-sweeping-map
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
#INSTALL( FILES ${tnl_fast_sweeping_map_SOURCES}
#         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/fast-sweeping-map )
