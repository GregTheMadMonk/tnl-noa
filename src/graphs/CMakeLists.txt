SET( headers tnlEllpackGraph.h
             tnlEllpackGraph_impl.h
             tnlEllpackGraphLinksAccessor.h
             tnlEllpackGraphLinksAccessor_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/graphs )
set( common_SOURCES
      )       

IF( BUILD_CUDA )
   set( tnl_graphs_CUDA__SOURCES
        ${common_SOURCES}      
        PARENT_SCOPE )
ENDIF()    

set( tnl_graphs_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/graphs )