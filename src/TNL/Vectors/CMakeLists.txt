set( headers Vector.h
             Vector_impl.h
             MultiVector.h
             MultiVector1D_impl.h
             MultiVector2D_impl.h
             MultiVector3D_impl.h
             MultiVector4D_impl.h
             VectorOperations.h
             VectorOperationsHost_impl.h
             VectorOperationsCuda_impl.h                                      
             SharedVector.h
             SharedVector_impl.h
             StaticVector.h 
             StaticVector_impl.h
             StaticVector1D_impl.h
             StaticVector2D_impl.h
             StaticVector3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Vectors )
set( common_SOURCES
     ${CURRENT_DIR}/VectorOperationsHost_impl.cpp
     ${CURRENT_DIR}/MultiVector_impl.cpp
     ${CURRENT_DIR}/SharedVector_impl.cpp
     ${CURRENT_DIR}/Vector_impl.cpp
     ${CURRENT_DIR}/StaticVector_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_vectors_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/VectorOperationsCuda_impl.cu
        ${CURRENT_DIR}/Vector_impl.cu
        ${CURRENT_DIR}/StaticVector_impl.cu 
        PARENT_SCOPE )
   set( tnl_vectors_SOURCES     
        ${common_SOURCES}
        PARENT_SCOPE )
else()
   set( tnl_vectors_SOURCES     
        ${common_SOURCES}
        ${CURRENT_DIR}/VectorOperationsCuda_impl.cpp
        PARENT_SCOPE )
ENDIF()    


INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Vectors )