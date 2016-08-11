set( headers Array.h
             Array_impl.h
             ArrayIO.h
             ArrayOperations.h
             ArrayOperationsHost_impl.h
             ArrayOperationsCuda_impl.h             
             MultiArray.h
             MultiArray1D_impl.h
             MultiArray2D_impl.h
             MultiArray3D_impl.h
             MultiArray4D_impl.h             
             SharedArray.h
             SharedArray_impl.h
             ConstSharedArray.h
             StaticArray.h 
             StaticArray_impl.h
             StaticArray1D_impl.h
             StaticArray2D_impl.h
             StaticArray3D_impl.h
             Vector.h
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
             StaticVector3D_impl.h  )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Containers )
set( common_SOURCES     
     ${CURRENT_DIR}/SharedArray_impl.cpp
     ${CURRENT_DIR}/MultiArray_impl.cpp
     ${CURRENT_DIR}/Array_impl.cpp
     ${CURRENT_DIR}/StaticArray_impl.cpp 
     ${CURRENT_DIR}/VectorOperationsHost_impl.cpp
     ${CURRENT_DIR}/MultiVector_impl.cpp
     ${CURRENT_DIR}/SharedVector_impl.cpp
     ${CURRENT_DIR}/Vector_impl.cpp
     ${CURRENT_DIR}/StaticVector_impl.cpp
)       

IF( BUILD_CUDA )
   set( tnl_containers_CUDA__SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/ArrayOperationsHost_impl.cu
        ${CURRENT_DIR}/ArrayOperationsCuda_impl.cu
        ${CURRENT_DIR}/Array_impl.cu
        ${CURRENT_DIR}/SharedArray_impl.cu
        ${CURRENT_DIR}/MultiArray_impl.cu
        ${CURRENT_DIR}/StaticArray_impl.cu
        ${CURRENT_DIR}/VectorOperationsCuda_impl.cu
        ${CURRENT_DIR}/Vector_impl.cu
        ${CURRENT_DIR}/StaticVector_impl.cu 
        PARENT_SCOPE )
ELSE()
   set( common_SOURCES
        ${common_SOURCES}
        ${CURRENT_DIR}/ArrayOperationsHost_impl.cpp
        ${CURRENT_DIR}/ArrayOperationsCuda_impl.cpp
 )               
ENDIF()    

set( tnl_containers_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
                   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Containers )