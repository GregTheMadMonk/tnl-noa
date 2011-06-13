SET( headers tnlDistributedGrid.h
             tnlGrid.h )

SET( sources ../dummy.cpp )   

SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/mesh )
SET( libtnlmeshinclude_HEADERS ${headers} )

ADD_LIBRARY( libtnlmesh${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( libtnlmesh${debugExt}-${tnlVersion} )

IF( BUILD_MPI )
    ADD_LIBRARY( libtnlmesh-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( libtnlmesh-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )