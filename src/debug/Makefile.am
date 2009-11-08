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

if BUILDMPI

lib_LTLIBRARIES = libtnldebug-0.1.la libtnldebug-mpi-0.1.la
libtnldebug_0_1_la_includedir = $(includedir)/tnldebug-0.1
libtnldebug_0_1_la_include_HEADERS = $(headers)
libtnldebug_0_1_la_SOURCES = $(sources)
libtnldebug_mpi_0_1_la_SOURCES = $(sources)
libtnldebug_mpi_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS)
libtnldebug_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)

else

lib_LTLIBRARIES = libtnldebug-0.1.la
libtnldebug_0_1_la_includedir = $(includedir)/tnldebug-0.1
libtnldebug_0_1_la_include_HEADERS = $(headers)
libtnldebug_0_1_la_SOURCES = $(sources)

endif

