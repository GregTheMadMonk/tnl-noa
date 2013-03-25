ADD_SUBDIRECTORY( krylov )
ADD_SUBDIRECTORY( stationary )

SET( headers tnlLinearResidueGetter_impl.h
   )
   
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/solvers/linear )
   
set( tnl_implementation_solvers_linear_stationary_SOURCES
     ${tnl_implementation_solvers_linear_krylov_SOURCES}
     ${tnl_implementation_solvers_linear_stationary_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA)
   set( tnl_implementation_solvers_linear_CUDA__SOURCES
        ${tnl_implementation_solvers_linear_krylov_CUDA__SOURCES}
        ${tnl_implementation_solvers_linear_stationary_CUDA__SOURCES}
        PARENT_SCOPE )
endif()
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/linear )