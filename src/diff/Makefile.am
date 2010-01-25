## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

headers = \
   mdiff-debug.h \
   drawGrid1D.h \
   drawGrid2D.h \
   drawGrid3D.h \
   tnlExplicitSolver.h \
   tnlGrid1D.h \
   tnlGrid2D.h \
   tnlGrid3D.h \
   tnlGridCUDA1D.h \
   tnlGridCUDA2D.h \
   tnlGridCUDA3D.h \
   tnlGridCUDA2DTester.h \
   tnlGridSystem1D.h \
   tnlEulerSolver.h \
   tnlMersonSolver.h \
   tnlMersonSolverCUDA.h \
   tnlMersonSolverCUDA.cu.h \
   tnlMPIMesh2D.h \
   tnlMPIMesh3D.h \
   tnlNonlinearRungeKuttaSolver.h \
   curve-ident.h \
   norms.h

sources =

if BUILD_CUDA
sources += tnlMersonSolverCUDA.cu
endif

libtnldiffincludedir = $(TNL_INCLUDE_DIR)/diff
libtnldiffinclude_HEADERS = $(headers)

noinst_LTLIBRARIES = libtnldiff-0.1.la
libtnldiff_0_1_la_SOURCES = $(sources) $(headers)
libtnldiff_0_1_la_CXXFLAGS = $(OPTIMISECXXFLAGS)

if BUILD_DBG
   noinst_LTLIBRARIES += libtnldiff-dbg-0.1.la 
   libtnldiff_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
   libtnldiff_dbg_0_1_la_LDFLAGS =  $(LDFLAGS) $(DBGLDFLAGS)
   libtnldiff_dbg_0_1_la_SOURCES =  $(sources) $(headers)
endif

if BUILD_MPI
   noinst_LTLIBRARIES += libtnldiff-mpi-0.1.la 
   libtnldiff_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMISECXXFLAGS)
   libtnldiff_mpi_0_1_la_LDFLAGS =  $(LDFLAGS) $(MPILDFLAGS)
   libtnldiff_mpi_0_1_la_SOURCES =  $(sources) $(headers)
endif


if BUILD_MPI_DBG
   noinst_LTLIBRARIES += libtnldiff-mpi-dbg-0.1.la 
   libtnldiff_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
   libtnldiff_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS) $(DBGLDFLAGS)
   libtnldiff_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
endif
