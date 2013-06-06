INCLUDE_DIRECTORIES( config )
ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( debug )
ADD_SUBDIRECTORY( diff )
ADD_SUBDIRECTORY( matrix )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( schemes )
ADD_SUBDIRECTORY( solvers )
ADD_SUBDIRECTORY( legacy )
ADD_SUBDIRECTORY( implementation )

set( tnl_SOURCES ${tnl_config_SOURCES}
                 ${tnl_core_SOURCES}
                 ${tnl_implementation_SOURCES}
                 ${tnl_legacy_SOURCES}
                 ${tnl_debug_SOURCES}
                 ${tnl_matrix_SOURCES} )

set( tnl_CUDA__SOURCES ${tnl_config_CUDA__SOURCES}
                       ${tnl_core_CUDA__SOURCES}
                       ${tnl_implementation_CUDA__SOURCES}
                       ${tnl_legacy_CUDA__SOURCES}
                       ${tnl_debug_CUDA__SOURCES}
                       ${tnl_matrix_CUDA__SOURCES} )
                 
                 
if( BUILD_CUDA )
   CUDA_ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED 
                     ${tnl_CUDA__SOURCES} )
else( BUILD_CUDA )
   ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED 
                ${tnl_SOURCES} )
endif( BUILD_CUDA )                                    
               
SET_TARGET_PROPERTIES( tnl${debugExt}-${tnlVersion} PROPERTIES 
                          SOVERSION 0 
                          VERSION ${tnlVersion} )
TARGET_LINK_LIBRARIES( tnl${debugExt}-${tnlVersion}  
                          ${BZIP2_LIBRARIES} )
INSTALL( TARGETS tnl${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
   
   if( BUILD_CUDA )
      CUDA_ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED
                        ${tnl_CUDA__SOURCES} )
   else( BUILD_CUDA )
         ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED
                      ${tnl_SOURCES} )  
   endif( BUILD_CUDA )                        
   
   SET_TARGET_PROPERTIES( tnl-mpi${debugExt}-${tnlVersion} PROPERTIES
                            SOVERSION 0 
                            VERSION ${tnlVersion} ) 
   
   TARGET_LINK_LIBRARIES( tnl-mpi${debugExt}-${tnlVersion} 
                            ${MPI_LIBRARIES} 
                            ${BZIP2_LIBRARIES} )
   
   INSTALL( TARGETS tnl-mpi${debugExt}-${tnlVersion} DESTINATION lib )
                                                                
ENDIF()


