set( tnl_hamilton_jacobi_SOURCES
     hamiltonJacobiProblemSetter.h
     hamiltonJacobiProblemSetter_impl.h
     hamiltonJacobiProblemSolver.h
     hamiltonJacobiProblemSolver_impl.h
     main.cpp
     hamiltonJacobiProblemConfig.h
     tnl-direct-eikonal-solver.h )
               
#ADD_EXECUTABLE(tnl-hamilton-jacobi main.cpp)
#target_link_libraries (tnl-hamilton-jacobi tnl )

ADD_EXECUTABLE(tnl-direct-eikonal-solver tnl-direct-eikonal-solver.cpp )
target_link_libraries (tnl-direct-eikonal-solver tnl )


INSTALL( TARGETS #tnl-hamilton-jacobi 
                 tnl-direct-eikonal-solver
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
