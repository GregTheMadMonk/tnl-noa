SET( headers tnlBICGSolver.h
             tnlBICGStabSolver.h
             tnlCGSolver.h
             tnlEulerSolver.h
             tnlExplicitSolver.h
             tnlGMRESSolver.h
             tnlILUPreconditioner.h
             tnlMatrixSolver.h
             tnlMersonSolver.h
             tnlPETSCSolver.h
             tnlPreconditioner.h
             tnlSORSolver.h
   )
   
SET( sources ../dummy.cpp )   

SET( libtnlsolverincludedir ${TNL_INCLUDE_DIR}/mesh )
SET( libtnlsolverinclude_HEADERS ${headers} )

ADD_LIBRARY( libtnlsolver${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( libtnlsolver${debugExt}-${tnlVersion} )

IF( BUILD_MPI )
    ADD_LIBRARY( libtnlsolver-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( libtnlsolver-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
ENDIF()


INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solver )
