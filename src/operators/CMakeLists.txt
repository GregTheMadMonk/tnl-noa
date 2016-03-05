ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )
ADD_SUBDIRECTORY( interpolants )
ADD_SUBDIRECTORY( operator-Q )
ADD_SUBDIRECTORY( operator-curvature )
#ADD_SUBDIRECTORY( godunov )
#ADD_SUBDIRECTORY( upwind )
ADD_SUBDIRECTORY( godunov-eikonal )
#ADD_SUBDIRECTORY( upwind-eikonal )

SET( headers tnlDirichletBoundaryConditions.h
             tnlDirichletBoundaryConditions_impl.h
             tnlExactFunctionInverseOperator.h
             tnlExactIdentityOperator.h
             tnlExactOperatorComposition.h
             tnlFiniteDifferences.h
             tnlFiniteDifferences_impl.h
             tnlFunctionInverseOperator.h
             tnlIdentityOperator.h
             tnlNeumannBoundaryConditions.h
             tnlNeumannBoundaryConditions_impl.h
             tnlOperatorComposition.h
             tnlOperator.h 
	     tnlNeumannReflectionBoundaryConditions.h
	     tnlNeumannReflectionBoundaryConditions_impl.h )
             
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/operators )

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
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators )
