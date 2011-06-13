INCLUDE_DIRECTORIES( config core debug diff matrix mesh solver )

ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( debug )
ADD_SUBDIRECTORY( diff )
ADD_SUBDIRECTORY( matrix )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( solver )


ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED dummy.cpp )
TARGET_LINK_LIBRARIES( tnl${debugExt}-${tnlVersion} libtnlconfig${debugExt}-${tnlVersion}
                                                    libtnlcore${debugExt}-${tnlVersion}
                                                    libtnldebug${debugExt}-${tnlVersion}
                                                    libtnldiff${debugExt}-${tnlVersion}
                                                    libtnlmatrix${debugExt}-${tnlVersion}
                                                    libtnlmesh${debugExt}-${tnlVersion}
                                                    libtnlsolver${debugExt}-${tnlVersion} )
INSTALL( TARGETS tnl${debugExt}-${tnlVersion} LIBRARY DESTINATION lib )

IF( BUILD_MPI )
  ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED dummy.cpp )
  TARGET_LINK_LIBRARIES( tnl-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}
                                                             libtnlconfig-mpi${debugExt}-${tnlVersion}
                                                             libtnlcore-mpi${debugExt}-${tnlVersion}
                                                             libtnldebug-mpi${debugExt}-${tnlVersion}
                                                             libtnldiff-mpi${debugExt}-${tnlVersion}
                                                             libtnlmatrix-mpi${debugExt}-${tnlVersion}
                                                             libtnlmesh-mpi${debugExt}-${tnlVersion}
                                                             libtnlsolver-mpi${debugExt}-${tnlVersion} )
   INSTALL( TARGETS tnl-mpi${debugExt}-${tnlVersion} LIBRARY DESTINATION lib )                                                             
ENDIF()


