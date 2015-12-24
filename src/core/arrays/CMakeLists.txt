set( headers tnlArray.h
             tnlArray_impl.h
             tnlArrayIO.h
             tnlArrayOperations.h
             tnlArrayOperationsHost_impl.h
             tnlArrayOperationsCuda_impl.h             
             tnlMultiArray.h
             tnlMultiArray1D_impl.h
             tnlMultiArray2D_impl.h
             tnlMultiArray3D_impl.h
             tnlMultiArray4D_impl.h             
             tnlSharedArray.h
             tnlSharedArray_impl.h
             tnlConstSharedArray.h
             tnlStaticArray.h 
             tnlStaticArray_impl.h
             tnlStaticArray1D_impl.h
             tnlStaticArray2D_impl.h
             tnlStaticArray3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core/arrays )
set( common_SOURCES     
#     ${CURRENT_DIR}/tnlSharedArray_impl.cpp
     ${CURRENT_DIR}/tnlMultiArray_impl.cpp
#     ${CURRENT_DIR}/tnlArray_impl.cpp
#     ${CURRENT_DIR}/tnlStaticArray_impl.cpp 
)       

IF( BUILD_CUDA )
   set( tnl_core_arrays_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/tnlArrayOperationsHost_impl.cu
        ${CURRENT_DIR}/tnlArrayOperationsCuda_impl.cu
#        ${CURRENT_DIR}/tnlArray_impl.cu
#        ${CURRENT_DIR}/tnlSharedArray_impl.cu
        ${CURRENT_DIR}/tnlMultiArray_impl.cu
#        ${CURRENT_DIR}/tnlStaticArray_impl.cu
        PARENT_SCOPE )
ELSE()
   set( common_SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/tnlArrayOperationsHost_impl.cpp
        ${CURRENT_DIR}/tnlArrayOperationsCuda_impl.cpp
 )               
ENDIF()    

set( tnl_core_arrays_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
                   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/arrays )