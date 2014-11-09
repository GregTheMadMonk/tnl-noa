ADD_SUBDIRECTORY( gradient )
ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )
ADD_SUBDIRECTORY( godunov )
ADD_SUBDIRECTORY( upwind )
ADD_SUBDIRECTORY( godunov-eikonal )
ADD_SUBDIRECTORY( upwind-eikonal )

SET( headers
     tnlDirichletBoundaryConditions_impl.h
     tnlFiniteDifferences_impl.h
     tnlNeumannBoundaryConditions_impl.h
     tnlNeumannReflectionBoundaryConditions_impl.h )
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/operators )

if( BUILD_CUDA)
      set( tnl_implementation_operators_CUDA__SOURCES        
        ${tnl_implementation_operators_diffusion_CUDA__SOURCES}
        ${tnl_implementation_operators_gradient_CUDA__SOURCES}
        ${tnl_implementation_operators_euler_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_operators_SOURCES     
     ${tnl_implementation_operators_diffusion_SOURCES}
     ${tnl_implementation_operators_gradient_SOURCES}
     ${tnl_implementation_operators_euler_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/operators )
