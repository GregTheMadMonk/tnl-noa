## Process this file with automake to produce Makefile.in

headers = tnlDbgDataElement.h \
	  tnlDebugEntry.h \
	  tnlDebugGroup.h \
	  tnlDebugParserbase.h \
	  tnlDebugParser.h \
          tnlDebug.h \
	  tnlDebugScanner.h \
	  tnlDebugStructure.h

sources = tnlDebug.cpp \
          tnlDebugGroup.cpp \
          tnlDebugStructure.cpp \
	  tnlDebugParser.cpp \
	  tnlDebugScanner.cpp \
	  parse.cc
          
bin_PROGRAMS = tnl-debug-test
tnl_debug_test_SOURCES = tnl-debug-test.cpp
tnl_debug_test_LDADD = libtnldebug-0.1.la 

lib_LTLIBRARIES = libtnldebug-0.1.la
libtnldebug_0_1_la_includedir = $(includedir)/tnldebug-0.1
libtnldebug_0_1_la_include_HEADERS = $(headers)
libtnldebug_0_1_la_SOURCES = $(sources) $(headers)

EXTRA_DIST = tnlDebugParser.y \
	     tnlDebugScanner.l \
             create-tnl-debug-parser
