ADD_SUBDIRECTORY( arrays )
ADD_SUBDIRECTORY( containers )
ADD_SUBDIRECTORY( cuda )
ADD_SUBDIRECTORY( vectors )
ADD_SUBDIRECTORY( images )
ADD_SUBDIRECTORY( multimaps )

set (headers tnlAssert.h               
             tnlConstants.h
             tnlCurve.h
             tnlCuda.h
             tnlCudaDeviceInfo.h
             tnlDataElement.h
             tnlDevice.h
             tnlDynamicTypeTag.h
             tnlFeature.h
             tnlFile.h 
             tnlFlopsCounter.h
             tnlHost.h 
             tnlIndexedSet.h
             tnlList.h
             tnlLogger.h 
             tnlObject.h 
             tnlStack.h
             tnlStaticFor.h
             tnlStatistics.h 
             tnlString.h 
             tnlReal.h
             tnlTimerCPU.h  
             tnlTimerRT.h    
             mfilename.h 
             mfuncs.h 
             mpi-supp.h 
             param-types.h
             tnlCuda_impl.h
             tnlLogger_impl.h 
             tnlIndexedSet_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core )
set( common_SOURCES
     ${CURRENT_DIR}/tnlTimerRT.cpp
     ${CURRENT_DIR}/tnlFile.cpp
     ${CURRENT_DIR}/tnlFlopsCounter.cpp 
     ${CURRENT_DIR}/tnlLogger.cpp 
     ${CURRENT_DIR}/tnlObject.cpp
     ${CURRENT_DIR}/tnlStatistics.cpp
     ${CURRENT_DIR}/tnlString.cpp 
     ${CURRENT_DIR}/tnlTimerCPU.cpp      
     ${CURRENT_DIR}/mfilename.cpp 
     ${CURRENT_DIR}/mpi-supp.cpp 
     ${CURRENT_DIR}/tnlCuda.cpp
     ${CURRENT_DIR}/tnlHost.cpp )       

IF( BUILD_CUDA )
   set( tnl_core_CUDA__SOURCES
        ${tnl_core_arrays_CUDA__SOURCES}
        ${tnl_core_containers_CUDA__SOURCES}
        ${tnl_core_cuda_CUDA__SOURCES}
        ${tnl_core_vectors_CUDA__SOURCES}
        ${tnl_core_images_CUDA__SOURCES}
        ${common_SOURCES} 
        ${CURRENT_DIR}/tnlCuda.cu
        ${CURRENT_DIR}/tnlCudaDeviceInfo.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_core_SOURCES     
     ${tnl_core_arrays_SOURCES}
     ${tnl_core_containers_SOURCES}
     ${tnl_core_cuda_SOURCES}
     ${tnl_core_vectors_SOURCES}
     ${tnl_core_images_SOURCES}
     ${common_SOURCES}
     ${CURRENT_DIR}/tnlCudaDeviceInfo.cpp
     PARENT_SCOPE )
    

SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlcoreinclude_HEADERS ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core )



