ADD_SUBDIRECTORY( containers )
ADD_SUBDIRECTORY( cuda )
ADD_SUBDIRECTORY( images )
ADD_SUBDIRECTORY( multimaps )

set (headers 
             tnlConstants.h
             tnlCurve.h
             tnlDynamicTypeTag.h
             tnlFeature.h
             tnlFlopsCounter.h
             tnlIndexedSet.h
             tnlReal.h
             mfilename.h 
             mfuncs.h 
             mpi-supp.h 
             param-types.h
             tnlIndexedSet_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/core )
set( common_SOURCES
     ${CURRENT_DIR}/tnlFlopsCounter.cpp 
     ${CURRENT_DIR}/mfilename.cpp 
     ${CURRENT_DIR}/mpi-supp.cpp )

IF( BUILD_CUDA )
   set( tnl_core_CUDA__SOURCES
        ${tnl_core_arrays_CUDA__SOURCES}
        ${tnl_core_containers_CUDA__SOURCES}
        ${tnl_core_cuda_CUDA__SOURCES}
        ${tnl_core_vectors_CUDA__SOURCES}
        ${tnl_core_images_CUDA__SOURCES}
        ${common_SOURCES} 
        PARENT_SCOPE )
ENDIF()    

set( tnl_core_SOURCES     
     ${tnl_core_arrays_SOURCES}
     ${tnl_core_containers_SOURCES}
     ${tnl_core_cuda_SOURCES}
     ${tnl_core_vectors_SOURCES}
     ${tnl_core_images_SOURCES}
     ${common_SOURCES}
     PARENT_SCOPE )
    

SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlcoreinclude_HEADERS ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core )



