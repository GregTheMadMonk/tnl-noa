ADD_SUBDIRECTORY( Arithmetics )
ADD_SUBDIRECTORY( Config )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Devices )
ADD_SUBDIRECTORY( Functions )
ADD_SUBDIRECTORY( Images )
ADD_SUBDIRECTORY( Matrices )
ADD_SUBDIRECTORY( Meshes )
ADD_SUBDIRECTORY( Operators )
ADD_SUBDIRECTORY( Problems )
ADD_SUBDIRECTORY( Solvers )

ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( legacy )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL )

set( headers 
     Assert.h
     Curve.h
     DevicePointer.h
     File.h
     File_impl.h
     Object.h
     List.h
     List_impl.h
     Logger.h
     Logger_impl.h
     SharedPointer.h
     SmartPointer.h
     SmartPointersRegister.h
     StaticFor.h
     String.h
     SystemInfo.h
     Timer.h
     TimerCPU.h
     TimerRT.h )

set( common_SOURCES 
     File.cpp
     Object.cpp
     Logger.cpp
     SmartPointersRegister.cpp
     String.cpp
     SystemInfo.cpp
     Timer.cpp
     TimerCPU.cpp
     TimerRT.cpp )
     
set( tnl_SOURCES ${tnl_config_SOURCES}
                 ${tnl_containers_SOURCES}
                 ${tnl_devices_SOURCES}
                 ${tnl_functions_SOURCES}
                 ${tnl_images_SOURCES}
                 ${tnl_matrices_SOURCES}
                 ${tnl_operators_SOURCES}
                 ${tnl_problems_SOURCES}
                 ${tnl_solvers_SOURCES}

                 ${tnl_core_SOURCES}
                 ${tnl_legacy_SOURCES}                 
                                                   
                 ${common_SOURCES} )

set( tnl_CUDA__SOURCES ${tnl_config_CUDA__SOURCES}
                       ${tnl_containers_CUDA__SOURCES}
                       ${tnl_devices_CUDA__SOURCES}
                       ${tnl_functions_CUDA__SOURCES}
                       ${tnl_images_CUDA__SOURCES}
                       ${tnl_matrices_CUDA__SOURCES}
                       ${tnl_operators_CUDA__SOURCES}
                       ${tnl_problems_CUDA__SOURCES}
                       ${tnl_solvers_CUDA__SOURCES}


                       ${tnl_core_CUDA__SOURCES}
                       ${tnl_legacy_CUDA__SOURCES}
                       ${common_SOURCES} )
                 
                 
if( BUILD_CUDA )
   CUDA_ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED ${tnl_CUDA__SOURCES}
                                                  OPTIONS ${CUDA_ADD_LIBRARY_OPTIONS} )
    if( HAVE_CUBLAS )
       CUDA_ADD_CUBLAS_TO_TARGET( tnl${debugExt}-${tnlVersion} )
    endif( HAVE_CUBLAS )
else( BUILD_CUDA )
   ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED 
                ${tnl_SOURCES} )
endif( BUILD_CUDA )                                    
               
SET_TARGET_PROPERTIES( tnl${debugExt}-${tnlVersion} PROPERTIES 
                       SOVERSION 0 
                       VERSION ${tnlVersion} )
TARGET_LINK_LIBRARIES( tnl${debugExt}-${tnlVersion} 
                       ${DCMTK_LIBRARIES})


INSTALL( TARGETS tnl${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
   
   if( BUILD_CUDA )
      CUDA_ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED ${tnl_CUDA__SOURCES} 
                                                         OPTIONS ${CUDA_ADD_LIBRARY_OPTIONS} )
      if( HAVE_CUBLAS )
         CUDA_ADD_CUBLAS_TO_TARGET( tnl-mpi${debugExt}-${tnlVersion} )
      endif( HAVE_CUBLAS )
   else( BUILD_CUDA )
         ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED
                      ${tnl_SOURCES} )  
   endif( BUILD_CUDA )                        
   
   SET_TARGET_PROPERTIES( tnl-mpi${debugExt}-${tnlVersion} PROPERTIES
                          SOVERSION 0 
                          VERSION ${tnlVersion} ) 
#   SET_TARGET_PROPERTIES( tnl-mpi${debugExt}-${tnlVersion} 
#                          LINK_INTERFACE_LIBRARIES "")

   
   TARGET_LINK_LIBRARIES( tnl-mpi${debugExt}-${tnlVersion} 
                          ${MPI_LIBRARIES} )
   INSTALL( TARGETS tnl-mpi${debugExt}-${tnlVersion} DESTINATION lib )
                                                                
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL )


