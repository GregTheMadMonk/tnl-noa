## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

EXTRA_DIST = tnlConfigDescriptionParser.y \
	         tnlConfigDescriptionScanner.l \
	         FlexLexer.h

headers =  \
		  tnlConfigDescription.h \
		  tnlCurve.h \
		  tnlDataElement.h \
		  tnlField1D.h \
		  tnlField2D.h \
		  mField3D.h \
		  mFieldSystem1D.h \
		  tnlList.h \
		  mLongVector.h \
		  tnlLogger.h \
		  tnlObject.h \
		  tnlString.h \
		  tnlConfigDescriptionParser.h \
		  tnlConfigDescriptionParserbase.h \
		  tnlConfigDescriptionParser.ih \
		  tnlConfigDescriptionScanner.h \
		  tnlParameterContainer.h \
		  tnlTimerCPU.h \
		  tnlTimerRT.h \
		  mVector.h \
		  compress-file.h \
		  mfilename.h \
		  mfuncs.h \
		  mpi-supp.h \
		  param-types.h

sources = tnlConfigDescription.cpp \
	  tnlLogger.cpp \
	  tnlObject.cpp \
	  tnlString.cpp \
	  tnlConfigDescriptionScanner.cpp \
	  tnlConfigDescriptionParser.cpp \
	  tnlParameterContainer.cpp \
	  tnlTimerCPU.cpp \
	  tnlTimerRT.cpp \
	  compress-file.cpp \
	  mfilename.cpp \
	  mpi-supp.cpp \
	  parse.cc

libmcoreincludedir = $(TNL_INCLUDE_DIR)/core
libmcoreinclude_HEADERS = $(headers)


noinst_LTLIBRARIES = libtnlcore-0.1.la
libtnlcore_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_0_1_la_CXXFLAGS = $(CXXFLAGS) $(OPTIMSECXXFLAGS)

if BUILD_MPI
noinst_LTLIBRARIES += libtnlcore-mpi-0.1.la
libtnlcore_mpi_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_mpi_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS) $(OPTIMSECXXFLAGS)
libtnlcore_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_DBG
noinst_LTLIBRARIES += libtnlcore-dbg-0.1.la
libtnlcore_dbg_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_dbg_0_1_la_CXXFLAGS = $(CXXFLAGS) $(DBGCXXFLAGS)
libtnlcore_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI_DBG
noinst_LTLIBRARIES += libtnlcore-mpi-dbg-0.1.la
libtnlcore_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_mpi_dbg_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS) $(DBGCXXFLAGS)
libtnlcore_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPICXXFLAGS) $(DBGLDFLAGS)
endif
