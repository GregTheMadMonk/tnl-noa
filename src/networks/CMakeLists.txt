SET( headers tnlEllpackGraph.h
             tnlEllpackGraph_impl.h
             tnlEllpackGraphLinksAccessor.h
             tnlEllpackGraphLinksAccessor_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/networks )
set( common_SOURCES
      )       

IF( BUILD_CUDA )
   set( tnl_networks_CUDA__SOURCES
        ${common_SOURCES}      
        PARENT_SCOPE )
ENDIF()    

set( tnl_networks_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/networks )