SET( headers  )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/operators/interpolants )

if( BUILD_CUDA)
      set( tnl_operators_interpolants_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_operators_interpolants_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )   
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/interpolants )
