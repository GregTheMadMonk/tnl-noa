## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

EXTRA_DIST = tnlConfigDescriptionParser.y \
	         tnlConfigDescriptionScanner.l \
	         FlexLexer.h

headers = tnlAssert.h \
          tnlConfigDescription.h \
    	    tnlCurve.h \
		    tnlDataElement.h \
		    tnlField1D.h \
		    tnlFieldCUDA1D.h \
		    tnlField2D.h \
		    tnlFieldCUDA2D.h \
		    tnlField3D.h \
		    tnlFieldCUDA3D.h \
		    tnlFieldSystem1D.h \
		    tnlList.h \
		    tnlLongVector.h \
		    tnlLongVectorCUDA.h \
		    tnlLogger.h \
		    tnlObject.h \
		    tnlStack.h \
		    tnlString.h \
		    tnlConfigDescriptionParser.h \
		    tnlConfigDescriptionParserbase.h \
		    tnlConfigDescriptionParser.ih \
		    tnlConfigDescriptionScanner.h \
		    tnlParameterContainer.h \
		    tnlTimerCPU.h \
		    tnlTimerRT.h \
		    tnlTester.h \
		    tnlVector.h \
		    tnl-cuda-kernels.h \
		    tnlCUDAKernelsTester.h \
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
	  tnlTester.cpp \
	  tnl-cuda-kernels.cu \
	  compress-file.cpp \
	  mfilename.cpp \
	  mpi-supp.cpp \
	  parse.cc

libmcoreincludedir = $(TNL_INCLUDE_DIR)/core
libmcoreinclude_HEADERS = $(headers)


noinst_LTLIBRARIES = libtnlcore-0.1.la
libtnlcore_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_0_1_la_CXXFLAGS = $(OPTIMSECXXFLAGS)

if BUILD_MPI
noinst_LTLIBRARIES += libtnlcore-mpi-0.1.la
libtnlcore_mpi_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMSECXXFLAGS)
libtnlcore_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_DBG
noinst_LTLIBRARIES += libtnlcore-dbg-0.1.la
libtnlcore_dbg_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
libtnlcore_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI_DBG
noinst_LTLIBRARIES += libtnlcore-mpi-dbg-0.1.la
libtnlcore_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnlcore_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
libtnlcore_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPICXXFLAGS) $(DBGLDFLAGS)
endif

libcore_tests_sources = tnlStringTester.cpp \
                        tnlStringTester.h \
                        tnlObjectTester.cpp \
                        tnlObjectTester.h \
                        tnlLongVectorCUDATester.cu \
                        tnlLongVectorCUDATester.cu.h \
                        tnlLongVectorCUDATester.h
                        

check_LTLIBRARIES = libcore-tests.la
libcore_tests_la_SOURCES = $(libcore_tests_sources)

if BUILD_DBG
noinst_LTLIBRARIES += libcore-tests-dbg.la
libcore_tests_dbg_la_SOURCES = $(libcore_tests_sources)
endif 
