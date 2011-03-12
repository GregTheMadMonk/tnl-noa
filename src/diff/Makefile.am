## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

headers = \
   tnlMPIMesh.h \
   tnlMPIMesh2D.h \
   tnlMPIMesh3D.h \
   curve-ident.h

sources =

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

libdiff_tests_sources = 

if BUILD_CUDA
libdiff_tests_sources += 
endif                        

check_LTLIBRARIES = libdiff-tests.la
libdiff_tests_la_SOURCES = $(libdiff_tests_sources)

if BUILD_DBG
noinst_LTLIBRARIES += libdiff-tests-dbg.la
libdiff_tests_dbg_la_SOURCES = $(libdiff_tests_sources)
endif 
