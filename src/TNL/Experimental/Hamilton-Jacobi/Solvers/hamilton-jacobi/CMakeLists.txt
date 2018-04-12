set( tnl_hamilton_jacobi_SOURCES
     hamiltonJacobiProblemSetter.h
     hamiltonJacobiProblemSetter_impl.h
     hamiltonJacobiProblemSolver.h
     hamiltonJacobiProblemSolver_impl.h
     main.cpp
     hamiltonJacobiProblemConfig.h
     tnl-direct-eikonal-solver.h )
               
#ADD_EXECUTABLE(tnl-hamilton-jacobi${debugExt} main.cpp)
#target_link_libraries (tnl-hamilton-jacobi${debugExt} tnl${debugExt}-${tnlVersion} )

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE(tnl-direct-eikonal-solver${debugExt} tnl-direct-eikonal-solver.cu )
    target_link_libraries (tnl-direct-eikonal-solver${debugExt} tnl${debugExt}-${tnlVersion} ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
    ADD_EXECUTABLE(tnl-direct-eikonal-solver${debugExt} tnl-direct-eikonal-solver.cpp )
    target_link_libraries (tnl-direct-eikonal-solver${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS #tnl-hamilton-jacobi${debugExt} 
                 tnl-direct-eikonal-solver${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
