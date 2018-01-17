set( tnl_hamilton_jacobi_parallel_SOURCES
#     MainBuildConfig.h
#     tnlParallelEikonalSolver2D_impl.h
#     tnlParallelEikonalSolver3D_impl.h
#     tnlParallelEikonalSolver.h
#     parallelEikonalConfig.h 
     main.cpp)


IF(  BUILD_CUDA ) 
	CUDA_ADD_EXECUTABLE(hamilton-jacobi-parallel main.cu)
ELSE(  BUILD_CUDA )                
	ADD_EXECUTABLE(hamilton-jacobi-parallel main.cpp)
ENDIF( BUILD_CUDA )
target_link_libraries (hamilton-jacobi-parallel tnl )


INSTALL( TARGETS hamilton-jacobi-parallel
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
#INSTALL( FILES ${tnl_hamilton_jacobi_parallel_SOURCES}
#         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/hamilton-jacobi-parallel )
