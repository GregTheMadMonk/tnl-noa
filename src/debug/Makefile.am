## Process this file with automake to produce Makefile.in

EXTRA_DIST = tnlDebugParser.y \
	     tnlDebugScanner.l \
             create-tnl-debug-parser
           
sources = tnlDebug.cpp \
          tnlDebugGroup.cpp \
          tnlDebugStructure.cpp \
	  tnlDebugParser.cpp \
	  tnlDebugScanner.cpp \
	  parse.cc

headers = tnlDbgDataElement.h \
	  tnlDebugEntry.h \
	  tnlDebugGroup.h \
	  tnlDebugParserbase.h \
	  tnlDebugParser.h \
          tnlDebug.h \
	  tnlDebugScanner.h \
	  tnlDebugStructure.h

bin_PROGRAMS = tnl-debug-test
tnl_debug_test_SOURCES = tnl-debug-test.cpp
tnl_debug_test_LDADD = libtnldebug-0.1.la 

lib_LTLIBRARIES = libtnldebug-0.1.la
libtnldebug_0_1_la_includedir = $(includedir)/tnldebug-0.1
libtnldebug_0_1_la_include_HEADERS = $(headers)
libtnldebug_0_1_la_SOURCES = $(sources)
libtnldebug_0_1_la_CXXFLAGS = $(CXXFLAGS) $(OPTIMISECXXFLAGS)

if BUILD_MPI
lib_LTLIBRARIES += libtnldebug-mpi-0.1.la
libtnldebug_mpi_0_1_la_SOURCES = $(sources)
libtnldebug_mpi_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS)
libtnldebug_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_DBG
lib_LTLIBRARIES += libtnldebug-dbg-0.1.la
libtnldebug_dbg_0_1_la_SOURCES = $(sources)
libtnldebug_dbg_0_1_la_CXXFLAGS = $(CXXFLAGS) $(DBGCXXFLAGS)
endif

if BUILD_MPI_DBG
lib_LTLIBRARIES += libtnldebug-mpi-dbg-0.1.la
libtnldebug_mpi_dbg_0_1_la_SOURCES = $(sources)
libtnldebug_mpi_dbg_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS) $(DBGCXXFLAGS)
libtnldebug_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif


