ADD_SUBDIRECTORY( krylov )
ADD_SUBDIRECTORY( stationary )

SET( headers tnlLinearResidueGetter.h
             tnlLinearResidueGetter_impl.h 
             tnlUmfpackWrapper.h
             tnlUmfpackWrapper_impl.h
   )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/solvers/linear )
   
set( tnl_solvers_linear_SOURCES
     ${tnl_solvers_linear_krylov_SOURCES}
     ${tnl_solvers_linear_stationary_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA)
   set( tnl_solvers_linear_CUDA__SOURCES
        ${tnl_solvers_linear_krylov_CUDA__SOURCES}
        ${tnl_solvers_linear_stationary_CUDA__SOURCES}
        PARENT_SCOPE )
endif()

   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear )
