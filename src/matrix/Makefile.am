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
libtnlmatrix_0_1_la_CXXFLAGS = $(MCORE_CFLAGS)
libtnlmatrix_0_1_la_LDFLAGS = $(MCORE_LIBS)
libtnlmatrix_0_1_la_SOURCES = $(sources) $(headers)

EXTRA_DIST = Doxyfile

