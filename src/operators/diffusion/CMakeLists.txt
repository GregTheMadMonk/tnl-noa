ADD_SUBDIRECTORY( nonlinear-diffusion-operators )

SET( headers tnlLinearDiffusion.h
             tnlLinearDiffusion_impl.h
             tnlExactLinearDiffusion.h             
             tnlExactLinearDiffusion_impl.h
	     tnlOneSidedNonlinearDiffusion.h             
             tnlExactNonlinearDiffusion.h
             tnlOneSidedMeanCurvature.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/operators/diffusion )

if( BUILD_CUDA)
      set( tnl_schemes_diffusion_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_schemes_diffusion_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )    
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/diffusion )
