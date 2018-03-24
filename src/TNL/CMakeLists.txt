ADD_SUBDIRECTORY( Config )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Debugging )
ADD_SUBDIRECTORY( Devices )
ADD_SUBDIRECTORY( Exceptions )
ADD_SUBDIRECTORY( Experimental )
ADD_SUBDIRECTORY( Functions )
ADD_SUBDIRECTORY( Images )
ADD_SUBDIRECTORY( Matrices )
ADD_SUBDIRECTORY( Meshes )
ADD_SUBDIRECTORY( Operators )
ADD_SUBDIRECTORY( Problems )
ADD_SUBDIRECTORY( Solvers )

#ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( legacy )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL )

set( headers
     Assert.h
     Constants.h
     CudaSharedMemory.h
     CudaStreamPool.h
     Curve.h
     DevicePointer.h
     File.h
     File_impl.h
     FileName.h
     Object.h
     Logger.h
     Logger_impl.h
     Math.h
     mpi-supp.h
     ParallelFor.h
     param-types.h
     SharedPointer.h
     SmartPointer.h
     SmartPointersRegister.h
     StaticFor.h
     String.h
     Timer.h
     UniquePointer.h )

set( common_SOURCES
     File.cpp
     FileName.cpp
     Object.cpp
     Logger.cpp
     SmartPointersRegister.cpp
     String.cpp
     Timer.cpp )

set( tnl_SOURCES ${tnl_config_SOURCES}
                 ${tnl_containers_SOURCES}
                 ${tnl_devices_SOURCES}
                 ${tnl_experimental_SOURCES}
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
                       ${tnl_experimental_CUDA__SOURCES}
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
   CUDA_ADD_LIBRARY( tnl SHARED ${tnl_CUDA__SOURCES}
                     OPTIONS ${CUDA_ADD_LIBRARY_OPTIONS} )
   if( HAVE_CUBLAS )
      CUDA_ADD_CUBLAS_TO_TARGET( tnl )
   endif( HAVE_CUBLAS )
   CUDA_ADD_LIBRARY( tnl_static STATIC ${tnl_CUDA__SOURCES} )
else( BUILD_CUDA )
   ADD_LIBRARY( tnl SHARED ${tnl_SOURCES} )
   ADD_LIBRARY( tnl_static STATIC ${tnl_SOURCES} )
#ifMIC
  #TARGET_COMPILE_DEFINITIONS( tnl PUBLIC -DHAVE_MIC )
endif( BUILD_CUDA )

SET_TARGET_PROPERTIES( tnl PROPERTIES
                       VERSION ${tnlVersion} )
TARGET_LINK_LIBRARIES( tnl
                       ${DCMTK_LIBRARIES} )


INSTALL( TARGETS tnl tnl_static DESTINATION lib )

IF( BUILD_MPI )

   if( BUILD_CUDA )
      CUDA_ADD_LIBRARY( tnl-mpi SHARED ${tnl_CUDA__SOURCES}
                        OPTIONS ${CUDA_ADD_LIBRARY_OPTIONS} )
      if( HAVE_CUBLAS )
         CUDA_ADD_CUBLAS_TO_TARGET( tnl-mpi )
      endif( HAVE_CUBLAS )
      CUDA_ADD_LIBRARY( tnl-mpi_static STATIC ${tnl_CUDA__SOURCES} )
   else( BUILD_CUDA )
      ADD_LIBRARY( tnl-mpi SHARED ${tnl_SOURCES} )
      ADD_LIBRARY( tnl-mpi_static STATIC ${tnl_SOURCES} )
   endif( BUILD_CUDA )

   SET_TARGET_PROPERTIES( tnl-mpi PROPERTIES
                          VERSION ${tnlVersion} )
#   SET_TARGET_PROPERTIES( tnl-mpi
#                          LINK_INTERFACE_LIBRARIES "")


   TARGET_LINK_LIBRARIES( tnl-mpi
                          ${MPI_LIBRARIES} )
   INSTALL( TARGETS tnl-mpi tnl-mpi_static DESTINATION lib )

endif()

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY} )
