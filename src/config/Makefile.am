## Process this file with automake to produce Makefile.in

EXTRA_DIST = tnlConfigDescriptionParser.y \
             tnlConfigDescriptionScanner.l \
             create-config-description-parser \
             FlexLexer.h

headers = tnlConfigDescription.h \
          tnlConfigDescriptionParser.h \
          tnlConfigDescriptionParserbase.h \
          tnlConfigDescriptionParser.ih \
          tnlConfigDescriptionScanner.h \
          tnlParameterContainer.h

sources = tnlConfigDescription.cpp \
          tnlConfigDescriptionScanner.cpp \
          tnlConfigDescriptionParser.cpp \
          tnlParameterContainer.cpp \
          parse.cc


libtnlconfigincludedir = $(TNL_INCLUDE_DIR)/config
libtnlconfiginclude_HEADERS = $(headers)


noinst_LTLIBRARIES = libtnlconfig-0.1.la
libtnlconfig_0_1_la_SOURCES = $(sources) $(headers)
libtnlconfig_0_1_la_CXXFLAGS = $(OPTIMSECXXFLAGS)

if BUILD_MPI
noinst_LTLIBRARIES += libtnlconfig-mpi-0.1.la
libtnlconfig_mpi_0_1_la_SOURCES = $(sources) $(headers)
libtnlconfig_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMSECXXFLAGS)
libtnlconfig_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_DBG
noinst_LTLIBRARIES += libtnlconfig-dbg-0.1.la
libtnlconfig_dbg_0_1_la_SOURCES = $(sources) $(headers)
libtnlconfig_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
libtnlconfig_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI_DBG
noinst_LTLIBRARIES += libtnlconfig-mpi-dbg-0.1.la
libtnlconfig_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
libtnlconfig_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
libtnlconfig_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPICXXFLAGS) $(DBGLDFLAGS)
endif
