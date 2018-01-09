set( tnl_narrow_band_SOURCES
#     MainBuildConfig.h
#     tnlNarrowBand2D_impl.h
#     tnlNarrowBand.h
#     narrowBandConfig.h 
     main.cpp)


IF(  BUILD_CUDA ) 
	CUDA_ADD_EXECUTABLE(narrow-band${debugExt} main.cu)
ELSE(  BUILD_CUDA )                
	ADD_EXECUTABLE(narrow-band${debugExt} main.cpp)
ENDIF( BUILD_CUDA )
target_link_libraries (narrow-band${debugExt} tnl${debugExt} )


INSTALL( TARGETS narrow-band${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
#INSTALL( FILES ${tnl_narrow_band_SOURCES}
#         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/narrow-band )
