ADD_SUBDIRECTORY( gradient )
ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )
ADD_SUBDIRECTORY( operator-Q )
ADD_SUBDIRECTORY( operator-curvature )

SET( headers tnlFiniteDifferences.h
             tnlFiniteDifferences_impl.h
             tnlDirichletBoundaryConditions.h
             tnlDirichletBoundaryConditions_impl.h
             tnlAnalyticDirichletBoundaryConditions.h
             tnlAnalyticDirichletBoundaryConditions_impl.h
             tnlNeumannBoundaryConditions.h
             tnlNeumannBoundaryConditions_impl.h
             tnlAnalyticNeumannBoundaryConditions.h
             tnlAnalyticNeumannBoundaryConditions_impl.h
             tnlExactOperatorEvaluator.h
             tnlOperatorEnumerator.h
             tnlOperatorEnumerator_impl.h )
             
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/operators )

if( BUILD_CUDA)
      set( tnl_operators_CUDA__SOURCES        
        ${tnl_operators_diffusion_CUDA__SOURCES}
        ${tnl_operators_gradient_CUDA__SOURCES}
        ${tnl_operators_euler_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_operators_SOURCES     
     ${tnl_operators_diffusion_SOURCES}
     ${tnl_operators_gradient_SOURCES}
     ${tnl_operators_euler_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators )
