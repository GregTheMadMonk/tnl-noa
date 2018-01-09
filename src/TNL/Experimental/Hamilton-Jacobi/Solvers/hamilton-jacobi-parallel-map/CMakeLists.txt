set( tnl_hamilton_jacobi_parallel_map_SOURCES
#     MainBuildConfig.h
#     tnlParallelMapSolver2D_impl.h
#     tnlParallelMapSolver.h
#     parallelMapConfig.h 
#	  main.cu
     main.cpp)


IF(  BUILD_CUDA ) 
	CUDA_ADD_EXECUTABLE(hamilton-jacobi-parallel-map${debugExt} main.cu)
ELSE(  BUILD_CUDA )                
	ADD_EXECUTABLE(hamilton-jacobi-parallel-map${debugExt} main.cpp)
ENDIF( BUILD_CUDA )
target_link_libraries (hamilton-jacobi-parallel-map${debugExt} tnl${debugExt}-${tnlVersion} )


INSTALL( TARGETS hamilton-jacobi-parallel-map${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
#INSTALL( FILES ${tnl_hamilton_jacobi_parallel_map_SOURCES}
#         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/hamilton-jacobi-parallel-map )
