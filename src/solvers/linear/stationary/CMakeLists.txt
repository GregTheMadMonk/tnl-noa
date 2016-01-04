SET( headers tnlSORSolver.h
             tnlSORSolver_impl.h
             tnlJacobiSolver.h
   )
   
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/solvers/linear/stationary )    
set( common_SOURCES ${CURRENT_DIR}/tnlSORSolver_impl.cpp )

set( tnl_solvers_linear_stationary_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )
     
if( BUILD_CUDA)
   set( tnl_solvers_linear_stationary_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()    
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear/stationary )
