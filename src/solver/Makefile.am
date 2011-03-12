## Process this file with automake to produce Makefile.in

headers = tnlExplicitSolver.h \
          tnlEulerSolver.h \
          tnlMersonSolver.h \
          tnlMatrixSolver.h \
          tnlPETSCSolver.h \
          tnlSORSolver.h \
          tnlCGSolver.h \
          tnlBICGSolver.h \
          tnlBICGStabSolver.h \
          tnlGMRESSolver.h  \
          tnlPreconditioner.h \
          tnlILUPreconditioner.h \
          tnlPETSCPreconditioner.h
          

sources = 
           
if BUILD_CUDA
sources += 
headers +=  
endif

libtnlsolverincludedir = $(TNL_INCLUDE_DIR)/solver
libtnlsolverinclude_HEADERS = $(headers)

noinst_LTLIBRARIES = libtnlsolver-0.1.la
libtnlsolver_0_1_la_SOURCES = $(sources) $(headers)
libtnlsolver_0_1_la_CXXFLAGS = $(OPTIMISECXXFLAGS)

if BUILD_DBG
   noinst_LTLIBRARIES += libtnlsolver-dbg-0.1.la
   libtnlsolver_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlsolver_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
   libtnlsolver_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI
   noinst_LTLIBRARIES += libtnlsolver-mpi-0.1.la
   libtnlsolver_mpi_0_1_la_SOURCES = $(sources) $(headers)
   libtnlsolver_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMISECXXFLAGS)
   libtnlsolver_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_MPI_DBG
   noinst_LTLIBRARIES += libtnlsolver-mpi-dbg-0.1.la
   libtnlsolver_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlsolver_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
   libtnlsolver_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS) $(DBGLDFLAGS)
endif