ADD_SUBDIRECTORY( Advection )
ADD_SUBDIRECTORY( Analytic )
ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )
ADD_SUBDIRECTORY( interpolants )
ADD_SUBDIRECTORY( operator-Q )
ADD_SUBDIRECTORY( operator-curvature )


SET( headers DirichletBoundaryConditions.h
             ExactFunctionInverseOperator.h
             ExactIdentityOperator.h
             ExactOperatorComposition.h
             FiniteDifferences.h
             FiniteDifferences_impl.h
             FunctionInverseOperator.h
             IdentityOperator.h
             NeumannBoundaryConditions.h
             OperatorComposition.h
             Operator.h )
             
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators )

if( BUILD_CUDA)
      set( tnl_operators_CUDA__SOURCES        
        ${tnl_operators_diffusion_CUDA__SOURCES}
        ${tnl_operators_gradient_CUDA__SOURCES}
        ${tnl_operators_euler_CUDA__SOURCES}
#        ${tnl_operators_godunov_CUDA__SOURCES}
        ${tnl_operators_godunov-eikonal_CUDA__SOURCES}
#        ${tnl_operators_upwind_CUDA__SOURCES}
#        ${tnl_operators_upwind-eikonal_CUDA__SOURCES}
        ${common_SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_operators_SOURCES     
     ${tnl_operators_diffusion_SOURCES}
     ${tnl_operators_gradient_SOURCES}
     ${tnl_operators_euler_SOURCES}
#     ${tnl_operators_godunov_SOURCES}
     ${tnl_operators_godunov-eikonal_SOURCES}
#     ${tnl_operators_upwind_SOURCES}
#     ${tnl_operators_upwind-eikonal_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Operators )
