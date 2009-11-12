## Process this file with automake to produce Makefile.in

headers = \
	  mCSRMatrix.h \
	  mFullMatrix.h \
	  mBaseMatrix.h \
	  mMatrixSolver.h \
	  mPETSCMatrix.h \
	  mPETSCSolver.h \
	  mSORSolver.h \
	  mCGSolver.h \
	  mBICGSolver.h \
	  mBICGStabSolver.h \
	  mGMRESSolver.h  \
	  mPreconditioner.h \
	  mILUPreconditioner.h \
	  mPETSCPreconditioner.h

sources = mBaseMatrix.cpp 

libtnlmatrixincludedir = $(includedir)/tnlmatrix-0.1

libtnlmatrixinclude_HEADERS = $(headers) mmatrix.h

lib_LTLIBRARIES = libtnlmatrix-0.1.la
libtnlmatrix_0_1_la_SOURCES = $(sources) $(headers)
libtnlmatrix_0_1_la_CXXFLAGS = $(CXXFLAGS) $(OPTIMISECXXFLAGS)

if BUILD_DBG
   lib_LTLIBRARIES += libtnlmatrix-dbg-0.1.la
   libtnlmatrix_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmatrix_dbg_0_1_la_CXXFLAGS =  $(CXXFLAGS) $(DBGCXXFLAGS)
   libtnlmatrix_dbg_0_1_la_LDFLAGS = $(LDFLAGS)    $(DBGLDFLAGS)
endif

if BUILD_MPI
   lib_LTLIBRARIES += libtnlmatrix-mpi-0.1.la
   libtnlmatrix_mpi_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmatrix_mpi_0_1_la_CXXFLAGS =  $(CXXFLAGS) $(MPICXXFLAGS) $(OPTIMISECXXFLAGS)
   libtnlmatrix_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_MPI_DBG
   lib_LTLIBRARIES += libtnlmatrix-mpi-dbg-0.1.la
   libtnlmatrix_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmatrix_mpi_dbg_0_1_la_CXXFLAGS =  $(CXXFLAGS) $(MPICXXFLAGS) $(DBGCXXFLAGS)
   libtnlmatrix_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS) $(DBGLDFLAGS)
endif


EXTRA_DIST = Doxyfile

