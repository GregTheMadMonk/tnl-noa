set( tnl_hamilton_jacobi_SOURCES
     hamiltonJacobiProblemSetter.h
     hamiltonJacobiProblemSetter_impl.h
     hamiltonJacobiProblemSolver.h
     hamiltonJacobiProblemSolver_impl.h
     main.cpp
     hamiltonJacobiProblemConfig.h )
               
ADD_EXECUTABLE(tnl-hamilton-jacobi${debugExt} main.cpp)
target_link_libraries (tnl-hamilton-jacobi${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS tnl-hamilton-jacobi${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
