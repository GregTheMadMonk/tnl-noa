set( headers tnlLaxFridrichs_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/schemes/euler/fvm )

if( BUILD_CUDA)
      set( tnl_implementation_schemes_euler_fvm_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_schemes_euler_fvm_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/schemes/euler/fvm )
