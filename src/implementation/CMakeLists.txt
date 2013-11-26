ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( generators )
ADD_SUBDIRECTORY( schemes )
ADD_SUBDIRECTORY( solvers )

SET( headers  )

IF( BUILD_CUDA )
   set( tnl_implementation_CUDA__SOURCES        
        ${tnl_implementation_core_CUDA__SOURCES}
        ${tnl_implementation_mesh_CUDA__SOURCES}
        ${tnl_implementation_generators_CUDA__SOURCES}
        ${tnl_implementation_schemes_CUDA__SOURCES}
        ${tnl_implementation_solvers_CUDA__SOURCES}         
        PARENT_SCOPE )
ENDIF() 

set( tnl_implementation_SOURCES 
     ${tnl_implementation_config_SOURCES}
     ${tnl_implementation_core_SOURCES}
     ${tnl_implementation_generators_SOURCES}
     ${tnl_implementation_mesh_SOURCES}
     ${tnl_implementation_schemes_SOURCES}
     ${tnl_implementation_solvers_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation )

