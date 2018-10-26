set (headers Cuda.h
             Cuda_impl.h
             CudaCallable.h
             CudaDeviceInfo.h
             Host.h
             MIC.h
             SystemInfo.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Devices )
set( common_SOURCES
     ${CURRENT_DIR}/MIC.cpp
     ${CURRENT_DIR}/SystemInfo.cpp )

IF( BUILD_CUDA )
   set( tnl_devices_CUDA__SOURCES
        ${common_SOURCES} 
        ${CURRENT_DIR}/CudaDeviceInfo.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_devices_SOURCES     
     ${common_SOURCES}
     ${CURRENT_DIR}/CudaDeviceInfo.cpp
     PARENT_SCOPE )
    
#SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
#SET( libtnlcoreinclude_HEADERS ${headers} )
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Devices )
