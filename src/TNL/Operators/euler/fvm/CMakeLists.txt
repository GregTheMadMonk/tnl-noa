SET( headers LaxFridrichs.h
             LaxFridrichs_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/schemes/euler/fvm )

if( BUILD_CUDA)
      set( tnl_schemes_euler_fvm_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_schemes_euler_fvm_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )
     
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/euler/fvm )
