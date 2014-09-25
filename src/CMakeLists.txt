INCLUDE_DIRECTORIES( config )
ADD_SUBDIRECTORY( functions )
ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( debug )
ADD_SUBDIRECTORY( matrices )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( operators )
ADD_SUBDIRECTORY( solvers )
ADD_SUBDIRECTORY( legacy )
ADD_SUBDIRECTORY( implementation )

set( tnl_SOURCES ${tnl_functions_SOURCES}
                 ${tnl_config_SOURCES}
                 ${tnl_core_SOURCES}
                 ${tnl_implementation_SOURCES}
                 ${tnl_legacy_SOURCES}
                 ${tnl_debug_SOURCES}
                 ${tnl_matrices_SOURCES}
                 ${tnl_operators_SOURCES} )

set( tnl_CUDA__SOURCES ${tnl_generators_CUDA__SOURCES}
                       ${tnl_config_CUDA__SOURCES}
                       ${tnl_core_CUDA__SOURCES}
                       ${tnl_implementation_CUDA__SOURCES}
                       ${tnl_legacy_CUDA__SOURCES}
                       ${tnl_debug_CUDA__SOURCES}
                       ${tnl_matrices_CUDA__SOURCES}
                       ${tnl_operators_CUDA__SOURCES} )
                 
                 
if( BUILD_CUDA )
   CUDA_ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED ${tnl_CUDA__SOURCES}
                                                  OPTIONS ${CUDA_ADD_LIBRARY_OPTIONS} )
else( BUILD_CUDA )
   ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED 
                ${tnl_SOURCES} )
endif( BUILD_CUDA )                                    
               
SET_TARGET_PROPERTIES( tnl${debugExt}-${tnlVersion} PROPERTIES 
                       SOVERSION 0 
                       VERSION ${tnlVersion} )

INSTALL( TARGETS tnl${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
   
   if( BUILD_CUDA )
      CUDA_ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED ${tnl_CUDA__SOURCES} 
                                                         OPTIONS ${CUDA_ADD_LIBRARY_OPTIONS} )
   else( BUILD_CUDA )
         ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED
                      ${tnl_SOURCES} )  
   endif( BUILD_CUDA )                        
   
   SET_TARGET_PROPERTIES( tnl-mpi${debugExt}-${tnlVersion} PROPERTIES
                          SOVERSION 0 
                          VERSION ${tnlVersion} ) 
   
   TARGET_LINK_LIBRARIES( tnl-mpi${debugExt}-${tnlVersion} 
                            ${MPI_LIBRARIES} )
   
   INSTALL( TARGETS tnl-mpi${debugExt}-${tnlVersion} DESTINATION lib )
                                                                
ENDIF()


