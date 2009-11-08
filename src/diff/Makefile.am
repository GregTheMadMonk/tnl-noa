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

libtnldiffincludedir = $(includedir)/tnldiff-0.1
libtnldiffinclude_HEADERS = $(headers)

if BUILDMPI

lib_LTLIBRARIES = libtnldiff-0.1.la  libtnldiff-mpi-0.1.la 
libtnldiff_0_1_la_CXXFLAGS = $(CXXFLAGS)
libtnldiff_0_1_la_LDFLAGS = $(LDFLAGS)
libtnldiff_0_1_la_SOURCES = $(sources) $(headers)
libtnldiff_mpi_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS)
libtnldiff_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
libtnldiff_mpi_0_1_la_SOURCES = $(sources) $(headers)

else

lib_LTLIBRARIES = libtnldiff-0.1.la
libtnldiff_0_1_la_SOURCES = $(sources) $(headers)

endif
