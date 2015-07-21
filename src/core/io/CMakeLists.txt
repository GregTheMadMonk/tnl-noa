set( headers  )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core/io )
set( common_SOURCES )       

IF( BUILD_CUDA )
   set( tnl_core_io_CUDA__SOURCES
        ${common_SOURCES}             
        PARENT_SCOPE )
ELSE()
    set( tnl_core_io_SOURCES     
         ${common_SOURCES}          
         PARENT_SCOPE )
ENDIF()    

        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/io )