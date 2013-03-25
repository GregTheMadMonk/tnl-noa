SET( headers tnlCGSolver_impl.h
             tnlBICGStabSolver_impl.h
             tnlGMRESSolver_impl.h
   )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers/linear/krylov )
set( common_SOURCES ${CURRENT_DIR}/tnlGMRESSolver_impl.cpp )     
set( tnl_implementation_solvers_linear_krylov_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA)
   set( tnl_implementation_solvers_linear_krylov_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif() 
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/linear/krylov )
