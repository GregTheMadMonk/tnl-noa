## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

headers = \
   mdiff.h \
   mdiff-debug.h \
   drawGrid1D.h \
   drawGrid2D.h \
   drawGrid3D.h \
   mExplicitSolver.h \
   mGrid1D.h \
   mGrid2D.h \
   mGrid3D.h \
   mGridSystem1D.h \
   mEulerSolver.h \
   mMersonSolver.h \
   mMPIMesh2D.h \
   mMPIMesh3D.h \
   mNonlinearRungeKuttaSolver.h \
   curve-ident.h \
   norms.h


libmdiffincludedir = $(includedir)/tnldiff-0.1
libmdiffinclude_HEADERS = $(headers)
lib_LTLIBRARIES = libtnldiff-0.1.la
libtnldiff_0_1_la_CXXFLAGS = $(mCore_CFLAGS)
libtnldiff_0_1_la_LDFLAGS = $(mCore_LIBS)
libtnldiff_0_1_la_SOURCES = $(sources) $(headers)


