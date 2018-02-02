SET( headers BackwardFiniteDifference.h
             CentralFiniteDifference.h
             ExactDifference.h
             tnlFiniteDifference.h
             tnlFiniteDifference_1D.h
             tnlFiniteDifference_2D.h
             tnlFiniteDifference_3D.h
             tnlForwardFinteDifference.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators/fdm )

if( BUILD_CUDA)
      set( tnl_operators_fdm_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_operators_fdm_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )   
   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/operators/fdm )
