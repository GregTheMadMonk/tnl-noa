## Process this file with automake to produce Makefile.in

headers = \
      tnlMatrix.h \
	  tnlCSRMatrix.h \
	  tnlFullMatrix.h \
	  tnlBaseMatrix.h \
	  tnlMatrixSolver.h \
	  tnlPETSCMatrix.h \
	  tnlPETSCSolver.h \
	  tnlSORSolver.h \
	  tnlCGSolver.h \
	  tnlBICGSolver.h \
	  tnlBICGStabSolver.h \
	  tnlGMRESSolver.h  \
	  tnlPreconditioner.h \
	  tnlILUPreconditioner.h \
	  tnlPETSCPreconditioner.h

sources = tnlBaseMatrix.cpp 

libtnlmatrixincludedir = $(TNL_INCLUDE_DIR)/matrix
libtnlmatrixinclude_HEADERS = $(headers)

noinst_LTLIBRARIES = libtnlmatrix-0.1.la
libtnlmatrix_0_1_la_SOURCES = $(sources) $(headers)
libtnlmatrix_0_1_la_CXXFLAGS = $(OPTIMISECXXFLAGS)

if BUILD_DBG
   noinst_LTLIBRARIES += libtnlmatrix-dbg-0.1.la
   libtnlmatrix_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmatrix_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
   libtnlmatrix_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI
   noinst_LTLIBRARIES += libtnlmatrix-mpi-0.1.la
   libtnlmatrix_mpi_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmatrix_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMISECXXFLAGS)
   libtnlmatrix_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_MPI_DBG
   noinst_LTLIBRARIES += libtnlmatrix-mpi-dbg-0.1.la
   libtnlmatrix_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmatrix_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
   libtnlmatrix_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS) $(DBGLDFLAGS)
endif


EXTRA_DIST = Doxyfile

