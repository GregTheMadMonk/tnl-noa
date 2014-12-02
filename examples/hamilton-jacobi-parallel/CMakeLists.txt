set( tnl_hamilton_jacobi_parallel_SOURCES
     MainBuildConfig.h
     tnlParallelEikonalSolver_impl.h
     tnlParallelEikonalSolver.h
     main.cpp
     parallelEikonalConfig.h )
               
ADD_EXECUTABLE(hamilton-jacobi-parallel${debugExt} main.cpp)
target_link_libraries (hamilton-jacobi-parallel${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS hamilton-jacobi-parallel${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES ${tnl_hamilton_jacobi_parallel_SOURCES}
               Makefile
         DESTINATION share/tnl-${tnlVersion}/examples/hamilton-jacobi-parallel )
