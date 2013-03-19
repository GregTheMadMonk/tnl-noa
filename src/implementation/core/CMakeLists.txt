SET( headers cuda-long-vector-kernels.h
             vector-operations.h
             memory-functions.h
             tnlArray_impl.h
             tnlLogger_impl.h
             tnlMultiArray1D_impl.h
             tnlMultiArray2D_impl.h
             tnlMultiArray3D_impl.h
             tnlMultiArray4D_impl.h
             tnlMultiVector1D_impl.h
             tnlMultiVector2D_impl.h
             tnlMultiVector3D_impl.h
             tnlMultiVector4D_impl.h            
             tnlSharedArray_impl.h
             tnlSharedVector_impl.h
             tnlVector_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core )
IF( BUILD_CUDA )
   set( tnl_implementation_core_CUDA_SOURCES
        ${CURRENT_DIR}/tnlArray_impl.cu
        ${CURRENT_DIR}/tnlVector_impl.cu )
ENDIF()    
set( tnl_implementation_core_SOURCES
     ${tnl_implementation_core_CUDA_SOURCES}
     ${CURRENT_DIR}/tnlArray_impl.cpp
     ${CURRENT_DIR}/tnlSharedArray_impl.cpp
     ${CURRENT_DIR}/tnlMultiArray_impl.cpp
     ${CURRENT_DIR}/tnlMultiVector_impl.cpp
     ${CURRENT_DIR}/tnlSharedVector_impl.cpp
     ${CURRENT_DIR}/tnlVector_impl.cpp
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
    PARENT_SCOPE )
    
    
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/core )

