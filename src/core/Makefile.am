## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

EXTRA_DIST = mConfigDescriptionParser.y \
	     mConfigDescriptionScanner.l \
	     FlexLexer.h

headers = mcore.h \
	  mConfigDescription.h \
	  mCurve.h \
	  mDataElement.h \
	  mField1D.h \
	  mField2D.h \
	  mField3D.h \
	  mFieldSystem1D.h \
	  mList.h \
	  mLongVector.h \
	  mLogger.h \
	  tnlObject.h \
	  mString.h \
	  mConfigDescriptionParser.h \
	  mConfigDescriptionParserbase.h \
	  mConfigDescriptionParser.ih \
	  mConfigDescriptionScanner.h \
	  mParameterContainer.h \
	  mTimerCPU.h \
	  mTimerRT.h \
	  mVector.h \
	  compress-file.h \
	  mfilename.h \
	  mfuncs.h \
	  mpi-supp.h \
	  param-types.h

sources = mConfigDescription.cpp \
	  mLogger.cpp \
	  tnlObject.cpp \
	  mString.cpp \
	  mConfigDescriptionScanner.cpp \
	  mConfigDescriptionParser.cpp \
	  mParameterContainer.cpp \
	  mTimerCPU.cpp \
	  mTimerRT.cpp \
	  compress-file.cpp \
	  mfilename.cpp \
	  mpi-supp.cpp \
	  parse.cc

libmcoreincludedir = $(includedir)/tnlcore-0.1
libmcoreinclude_HEADERS = $(headers)


if BUILDMPI

lib_LTLIBRARIES = libtnlcore-0.1.la libtnlcore-mpi-0.1.la
libtnlcore_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_mpi_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_mpi_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS)
libtnlcore_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)

else

lib_LTLIBRARIES = libtnlcore-0.1.la
libtnlcore_0_1_la_SOURCES = $(sources) $(headers) debug.h

endif
