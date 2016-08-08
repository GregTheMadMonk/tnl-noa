ADD_SUBDIRECTORY( Preconditioners )

SET( headers BICGStab.h
             BICGStab_impl.h
             CG.h
             CG_impl.h
             GMRES.h
             GMRES_impl.h
             LinearResidueGetter.h
             LinearResidueGetter_impl.h              
             SOR.h
             SOR_impl.h
             TFQMR.h
             TFQMR_impl.h
             UmfpackWrapper.h
             UmfpackWrapper_impl.h

   )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Solvers/Linear )

set( common_SOURCES ${CURRENT_DIR}/GMRES_impl.cpp
                    ${CURRENT_DIR}/SOR_impl.cpp )
   
set( tnl_solvers_linear_SOURCES
     ${tnl_solvers_linear_krylov_SOURCES}
     ${tnl_solvers_linear_preconditioners_SOURCES}
     ${tnl_solvers_linear_stationary_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA)
   set( tnl_solvers_linear_CUDA__SOURCES
        ${tnl_solvers_linear_krylov_CUDA__SOURCES}
        ${tnl_solvers_linear_preconditioners_CUDA_SOURCES}
        ${tnl_solvers_linear_stationary_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/Linear )
