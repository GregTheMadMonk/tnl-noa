SET( headers tnlCGSolver_impl.h
             tnlBICGStabSolver_impl.h
             tnlGMRESSolver_impl.h
   )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers/linear/krylov )    
set( tnl_implementation_solvers_linear_krylov_SOURCES
     ${CURRENT_DIR}/tnlGMRESSolver_impl.cpp
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/linear/krylov )
