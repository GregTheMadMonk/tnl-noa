SET( headers tnlFunctionDiscretizer_impl.h
             tnlSinWaveFunction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/generators/functions )
set( common_SOURCES
      )       

IF( BUILD_CUDA )
   set( tnl_implementation_generators_functions_CUDA__SOURCES
        ${common_SOURCES} 
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_generators_functions_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/generators/functions )

