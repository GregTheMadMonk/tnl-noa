SET( headers tnlContainer_impl.h
             tnlStaticContainer_impl.h
             tnlUniqueContainer_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core/containers )
set( common_SOURCES )       

IF( BUILD_CUDA )
   set( tnl_implementation_core_containers_CUDA__SOURCES
        ${common_SOURCES}             
        ${CURRENT_DIR}/tnlContainer_impl.cu
        ${CURRENT_DIR}/tnlStaticContainer_impl.cu
        PARENT_SCOPE )
ELSE()
   set( common_SOURCES
        ${common_SOURCES}
         )               
ENDIF()    

set( tnl_implementation_core_containers_SOURCES     
     ${common_SOURCES}          
     ${CURRENT_DIR}/tnlContainer_impl.cpp
     ${CURRENT_DIR}/tnlStaticContainer_impl.cpp
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/core/containers )

