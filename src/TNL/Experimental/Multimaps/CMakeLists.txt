SET( headers tnlEllpackIndexMultimap.h
             tnlEllpackIndexMultimap_impl.h
             tnlEllpackIndexMultimapValues.h
             tnlEllpackIndexMultimapValues_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/core/multimaps )
set( common_SOURCES
      )       

IF( BUILD_CUDA )
   set( tnl_core_multimaps_CUDA__SOURCES
        ${common_SOURCES}      
        PARENT_SCOPE )
ENDIF()    

set( tnl_core_multimaps_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/core/multimaps )
