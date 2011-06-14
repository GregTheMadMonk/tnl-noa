SET( headers tnlDistributedGrid.h
             tnlGrid.h )

SET( sources ../dummy.cpp )   

SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/mesh )
SET( libtnlmeshinclude_HEADERS ${headers} )

ADD_LIBRARY( tnlmesh${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( libtnlmesh${debugExt}-${tnlVersion} )
INSTALL( TARGETS tnlmesh${debugExt}-${tnlVersion} LIBRARY DESTINATION lib )

IF( BUILD_MPI )
    ADD_LIBRARY( tnlmesh-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( tnlmesh-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    SET_TARGET_PROPERTIES( tnlmesh-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    INSTALL( TARGETS tnlmesh-mpi${debugExt}-${tnlVersion} LIBRARY DESTINATION lib )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )