ADD_SUBDIRECTORY( functions )

SET( headers  )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/generators )
set( common_SOURCES
      )       

IF( BUILD_CUDA )
   set( tnl_implementation_generators_CUDA__SOURCES
        ${tnl_implementation_generators_functions_CUDA__SOURCES}
        ${common_SOURCES} 
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_generators_SOURCES     
     ${tnl_implementation_generators_functions_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/generators )

