SET( headers tnlMPIMesh.h
             tnlMPIMesh2D.h 
             tnlMPIMesh3D.h  
             curve-ident.h )
             
SET( sources ../dummy.cpp )             

SET( libtnldiffincludedir ${TNL_INCLUDE_DIR}/diff )
SET( libtnldiffinclude_HEADERS ${headers} )

ADD_LIBRARY( libtnldiff${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( libtnldiff${debugExt}-${tnlVersion} )

IF( BUILD_MPI )
    ADD_LIBRARY( libtnldiff-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( libtnldiff-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/diff )