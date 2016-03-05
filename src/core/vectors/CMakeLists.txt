set( headers tnlVector.h
             tnlVector_impl.h
             tnlMultiVector.h
             tnlMultiVector1D_impl.h
             tnlMultiVector2D_impl.h
             tnlMultiVector3D_impl.h
             tnlMultiVector4D_impl.h
             tnlVectorOperations.h
             tnlVectorOperationsHost_impl.h
             tnlVectorOperationsCuda_impl.h                                      
             tnlSharedVector.h
             tnlSharedVector_impl.h
             tnlStaticVector.h 
             tnlStaticVector_impl.h
             tnlStaticVector1D_impl.h
             tnlStaticVector2D_impl.h
             tnlStaticVector3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core/vectors )
set( common_SOURCES
     ${CURRENT_DIR}/tnlVectorOperationsHost_impl.cpp
     ${CURRENT_DIR}/tnlMultiVector_impl.cpp
     ${CURRENT_DIR}/tnlSharedVector_impl.cpp
     ${CURRENT_DIR}/tnlVector_impl.cpp
     ${CURRENT_DIR}/tnlStaticVector_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_core_vectors_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/tnlVectorOperationsCuda_impl.cu
        ${CURRENT_DIR}/tnlVector_impl.cu
        ${CURRENT_DIR}/tnlStaticVector_impl.cu 
        PARENT_SCOPE )
   set( tnl_core_vectors_SOURCES     
        ${common_SOURCES}
        PARENT_SCOPE )
else()
   set( tnl_core_vectors_SOURCES     
        ${common_SOURCES}
        ${CURRENT_DIR}/tnlVectorOperationsCuda_impl.cpp
        PARENT_SCOPE )
ENDIF()    


INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/vectors )