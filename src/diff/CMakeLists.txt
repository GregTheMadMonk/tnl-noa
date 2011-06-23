SET( headers tnlMPIMesh.h
             tnlMPIMesh2D.h 
             tnlMPIMesh3D.h  
             curve-ident.h )
             
SET( sources ../dummy.cpp )             

SET( libtnldiffincludedir ${TNL_INCLUDE_DIR}/diff )
SET( libtnldiffinclude_HEADERS ${headers} )

ADD_LIBRARY( tnldiff${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( tnldiff${debugExt}-${tnlVersion} )
INSTALL( TARGETS tnldiff${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
    ADD_LIBRARY( tnldiff-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( tnldiff-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    SET_TARGET_PROPERTIES( tnldiff-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    INSTALL( TARGETS tnldiff-mpi${debugExt}-${tnlVersion} DESTINATION lib )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/diff )