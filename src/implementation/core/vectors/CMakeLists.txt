SET( headers tnlMultiVector1D_impl.h
             tnlMultiVector2D_impl.h
             tnlMultiVector3D_impl.h
             tnlMultiVector4D_impl.h            
             tnlSharedVector_impl.h
             tnlVector_impl.h
             tnlVectorOperationsHost_impl.h
             tnlVectorOperationsCuda_impl.h
             tnlStaticVector_impl.h
             tnlStaticVector1D_impl.h
             tnlStaticVector2D_impl.h
             tnlStaticVector3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/core/vectors )
set( common_SOURCES
     ${CURRENT_DIR}/tnlVectorOperations_impl.cpp
     ${CURRENT_DIR}/tnlMultiVector_impl.cpp
     ${CURRENT_DIR}/tnlSharedVector_impl.cpp
     ${CURRENT_DIR}/tnlVector_impl.cpp
     ${CURRENT_DIR}/tnlStaticVector_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_implementation_core_vectors_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/tnlVectorOperations_impl.cu
        ${CURRENT_DIR}/tnlVector_impl.cu
        ${CURRENT_DIR}/tnlStaticVector_impl.cu 
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_core_vectors_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/core/vectors )

