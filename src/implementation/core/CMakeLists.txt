ADD_SUBDIRECTORY( arrays )
ADD_SUBDIRECTORY( containers )
ADD_SUBDIRECTORY( cuda )
ADD_SUBDIRECTORY( vectors )

SET( headers tnlLogger_impl.h 
             tnlIndexedSet_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core )
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
   set( tnl_implementation_core_CUDA__SOURCES
        ${tnl_implementation_core_arrays_CUDA__SOURCES}
        ${tnl_implementation_core_containers_CUDA__SOURCES}
        ${tnl_implementation_core_cuda_CUDA__SOURCES}
        ${tnl_implementation_core_vectors_CUDA__SOURCES}
        ${common_SOURCES} 
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_core_SOURCES     
     ${tnl_implementation_core_arrays_SOURCES}
     ${tnl_implementation_core_containers_SOURCES}
     ${tnl_implementation_core_cuda_SOURCES}
     ${tnl_implementation_core_vectors_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/core )

