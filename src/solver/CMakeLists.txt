SET( headers tnlBICGSolver.h
             tnlBICGStabSolver.h
             tnlCGSolver.h
             tnlEulerSolver.h
             tnlExplicitSolver.h
             tnlGMRESSolver.h
             tnlILUPreconditioner.h
             tnlMatrixSolver.h
             tnlMersonSolver.h
             tnlPETSCPreconditioner.h
             tnlPETSCSolver.h
             tnlPreconditioner.h
             tnlSORSolver.h
   )
   
SET( sources ../dummy.cpp )   

SET( libtnlsolverincludedir ${TNL_INCLUDE_DIR}/mesh )
SET( libtnlsolverinclude_HEADERS ${headers} )

ADD_LIBRARY( tnlsolver${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( tnlsolver${debugExt}-${tnlVersion} )
INSTALL( TARGETS tnlsolver${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
    ADD_LIBRARY( tnlsolver-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( tnlsolver-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    SET_TARGET_PROPERTIES( tnlsolver-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    INSTALL( TARGETS tnlsolver-mpi${debugExt}-${tnlVersion} DESTINATION lib )
ENDIF()


INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solver )
