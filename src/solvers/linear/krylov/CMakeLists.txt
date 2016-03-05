SET( headers tnlCGSolver.h
             tnlCGSolver_impl.h
             tnlBICGStabSolver.h
             tnlBICGStabSolver_impl.h
             tnlGMRESSolver.h
             tnlGMRESSolver_impl.h
             tnlTFQMRSolver.h
             tnlTFQMRSolver_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/solvers/linear/krylov )
set( common_SOURCES ${CURRENT_DIR}/tnlGMRESSolver_impl.cpp )
set( tnl_solvers_linear_krylov_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA)
   set( tnl_solvers_linear_krylov_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/linear/krylov )
