ADD_SUBDIRECTORY( matrices )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( functions )
ADD_SUBDIRECTORY( operators )
ADD_SUBDIRECTORY( solvers )

SET( headers  )

IF( BUILD_CUDA )
   set( tnl_implementation_CUDA__SOURCES        
        ${tnl_implementation_core_CUDA__SOURCES}
        ${tnl_implementation_matrices_CUDA__SOURCES}
        ${tnl_implementation_mesh_CUDA__SOURCES}
        ${tnl_implementation_functions_CUDA__SOURCES}
        ${tnl_implementation_operators_CUDA__SOURCES}
        ${tnl_implementation_solvers_CUDA__SOURCES}         
        PARENT_SCOPE )
ENDIF() 

set( tnl_implementation_SOURCES 
     ${tnl_implementation_core_SOURCES}
     ${tnl_implementation_functions_SOURCES}
     ${tnl_implementation_matrices_SOURCES}
     ${tnl_implementation_mesh_SOURCES}
     ${tnl_implementation_operators_SOURCES}
     ${tnl_implementation_solvers_SOURCES} 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation )

