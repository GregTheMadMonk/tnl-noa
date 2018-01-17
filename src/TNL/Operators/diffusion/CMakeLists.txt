SET( headers LinearDiffusion.h
             LinearDiffusion_impl.h
             ExactLinearDiffusion.h             
             ExactLinearDiffusion_impl.h
	     OneSidedNonlinearDiffusion.h             
             ExactNonlinearDiffusion.h
             OneSidedMeanCurvature.h
             FiniteVolumeNonlinearOperator.h
             FiniteVolumeNonlinearOperator_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators/diffusion )

if( BUILD_CUDA)
      set( tnl_schemes_diffusion_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_schemes_diffusion_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )    
   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Operators/diffusion )
