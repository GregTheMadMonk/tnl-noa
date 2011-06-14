INCLUDE_DIRECTORIES( config core debug diff matrix mesh solver )

ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( core )
ADD_SUBDIRECTORY( debug )
ADD_SUBDIRECTORY( diff )
ADD_SUBDIRECTORY( matrix )
ADD_SUBDIRECTORY( mesh )
ADD_SUBDIRECTORY( solver )


ADD_LIBRARY( tnl${debugExt}-${tnlVersion} SHARED dummy.cpp )
TARGET_LINK_LIBRARIES( tnl${debugExt}-${tnlVersion} tnlconfig${debugExt}-${tnlVersion}
                                                    tnlcore${debugExt}-${tnlVersion}
                                                    tnldebug${debugExt}-${tnlVersion}
                                                    tnldiff${debugExt}-${tnlVersion}
                                                    tnlmatrix${debugExt}-${tnlVersion}
                                                    tnlmesh${debugExt}-${tnlVersion}
                                                    tnlsolver${debugExt}-${tnlVersion} )
INSTALL( TARGETS tnl${debugExt}-${tnlVersion} LIBRARY DESTINATION lib )

IF( BUILD_MPI )
  ADD_LIBRARY( tnl-mpi${debugExt}-${tnlVersion} SHARED dummy.cpp )
  TARGET_LINK_LIBRARIES( tnl-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}
                                                             tnlconfig-mpi${debugExt}-${tnlVersion}
                                                             tnlcore-mpi${debugExt}-${tnlVersion}
                                                             tnldebug-mpi${debugExt}-${tnlVersion}
                                                             tnldiff-mpi${debugExt}-${tnlVersion}
                                                             tnlmatrix-mpi${debugExt}-${tnlVersion}
                                                             tnlmesh-mpi${debugExt}-${tnlVersion}
                                                             tnlsolver-mpi${debugExt}-${tnlVersion} )
   INSTALL( TARGETS tnl-mpi${debugExt}-${tnlVersion} LIBRARY DESTINATION lib )                                                             
ENDIF()


