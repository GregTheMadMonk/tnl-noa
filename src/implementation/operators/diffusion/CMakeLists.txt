set( headers tnlLinearDiffusion_impl.h
             tnlExactLinearDiffusion_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/operators/diffusion )

if( BUILD_CUDA)
      set( tnl_implementation_schemes_diffusion_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_schemes_diffusion_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/operators/diffusion )
