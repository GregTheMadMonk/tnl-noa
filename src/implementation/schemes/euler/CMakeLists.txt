ADD_SUBDIRECTORY( fvm )


SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/schemes/euler )

if( BUILD_CUDA)
      set( tnl_implementation_schemes_euler_CUDA__SOURCES
        ${tnl_implementation_schemes_euler_fvm_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_schemes_euler_SOURCES
     ${tnl_implementation_schemes_euler_fvm_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )     
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/schemes/euler )
