## Process this file with automake to produce Makefile.in

#if BUILDMPI
#SUBDIRS = mpi
#else
#SUBDIRS = 
#endif

SUBDIRS = low-level

headers = tnlAssert.h \
          tnlArray.h \
          tnlCommunicator.h \
          tnlCoordinates.h \
          tnlCriticalSection.h \
    	    tnlCurve.h \
    	    tnlCudaSupport.h \
		    tnlDataElement.h \
		    tnlFieldSystem1D.h \
		    tnlFile.h \
		    tnlFlopsCounter.h \
		    tnlList.h \
		    tnlLongVector.h \
		    tnlLongVectorBase.h \
		    tnlLongVectorHost.h \
		    tnlLongVectorCUDA.h \
		    tnlLogger.h \
		    tnlObject.h \
		    tnlSharedMemory.h \
		    tnlStack.h \
		    tnlStatistics.h \
		    tnlString.h \
		    tnlReal.h \
		    tnlTimerCPU.h \
		    tnlTimerRT.h \
		    tnlVector.h \
		    mfilename.h \
		    mfuncs.h \
		    mpi-supp.h \
		    param-types.h

sources = tnlCriticalSection.cpp \
          tnlCudaSupport.cpp \
          tnlFile.cpp \
          tnlFlopsCounter.cpp \
	       tnlLogger.cpp \
	       tnlObject.cpp \
	       tnlStatistics.cpp \
	       tnlString.cpp \
	       tnlTimerCPU.cpp \
	       tnlTimerRT.cpp \
	       mfilename.cpp \
	       mpi-supp.cpp 
	  
	  
	  
if BUILD_CUDA
sources +=
endif

libtnlcoreincludedir = $(TNL_INCLUDE_DIR)/core
libtnlcoreinclude_HEADERS = $(headers)


noinst_LTLIBRARIES = libtnlcore-0.1.la
libtnlcore_0_1_la_SOURCES = $(sources) $(headers)
libtnlcore_0_1_la_CXXFLAGS = $(OPTIMSECXXFLAGS)

if BUILD_MPI
noinst_LTLIBRARIES += libtnlcore-mpi-0.1.la
libtnlcore_mpi_0_1_la_SOURCES = $(sources) $(headers)
libtnlcore_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMSECXXFLAGS)
libtnlcore_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_DBG
noinst_LTLIBRARIES += libtnlcore-dbg-0.1.la
libtnlcore_dbg_0_1_la_SOURCES = $(sources) $(headers)
libtnlcore_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
libtnlcore_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI_DBG
noinst_LTLIBRARIES += libtnlcore-mpi-dbg-0.1.la
libtnlcore_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
libtnlcore_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
libtnlcore_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPICXXFLAGS) $(DBGLDFLAGS)
endif

check_LTLIBRARIES = libcore-tests.la
libcore_tests_la_SOURCES = $(libcore_tests_sources)

if BUILD_DBG
noinst_LTLIBRARIES += libcore-tests-dbg.la
libcore_tests_dbg_la_SOURCES = $(libcore_tests_sources)
endif 
