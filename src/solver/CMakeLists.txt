set( headers tnlBICGSolver.h
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
set( sources )

set( libtnlsolverincludedir ${TNL_INCLUDE_DIR}/mesh )
set( libtnlsolverinclude_HEADERS ${headers} )

ADD_LIBRARY(libtnlsolver${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers})    #${OPTIMISECXXFLAGS}

if( BUILD_MPI )
  target_link_libraries( libtnlsolver-0.1 mpi )
endif()


