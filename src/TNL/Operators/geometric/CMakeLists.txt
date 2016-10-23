SET( headers ExactGradientNorm.h
             FDMGradientNorm.h             
             TwoSidedGradientNorm.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators/geometric )

if( BUILD_CUDA)
      set( tnl_operators_geometric_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_operators_geometric_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )   
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/geometric )
