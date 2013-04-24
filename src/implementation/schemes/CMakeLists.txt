ADD_SUBDIRECTORY( euler )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/schemes )

if( BUILD_CUDA)
      set( tnl_implementation_schemes_CUDA__SOURCES
        ${tnl_implementation_schemes_euler_CUDA__SOURCES}
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_schemes_SOURCES
     ${tnl_implementation_schemes_euler_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/schemes )
