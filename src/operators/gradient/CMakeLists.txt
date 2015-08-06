SET( headers tnlCentralFDMGradient.h
             tnlCentralFDMGradient_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/schemes/gradient )

if( BUILD_CUDA)
      set( tnl_schemes_gradient_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_schemes_gradient_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )   
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/schemes/gradient )
