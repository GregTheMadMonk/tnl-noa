SET( headers tnlSORSolver_impl.h
   )
   
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers/linear/stationary )    
set( tnl_implementation_solvers_linear_stationary_SOURCES
     ${CURRENT_DIR}/tnlSORSolver_impl.cpp
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/linear/stationary )
