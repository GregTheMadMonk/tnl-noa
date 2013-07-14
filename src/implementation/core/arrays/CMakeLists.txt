SET( headers tnlArray_impl.h
             tnlMultiArray1D_impl.h
             tnlMultiArray2D_impl.h
             tnlMultiArray3D_impl.h
             tnlMultiArray4D_impl.h
             tnlSharedArray_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core/arrays )
set( common_SOURCES
     ${CURRENT_DIR}/tnlSharedArray_impl.cpp
     ${CURRENT_DIR}/tnlMultiArray_impl.cpp
     ${CURRENT_DIR}/tnlArray_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_implementation_core_arrays_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/tnlArray_impl.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_core_arrays_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/core/arrays )

