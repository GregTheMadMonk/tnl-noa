## Process this file with automake to produce Makefile.in

headers = \
   debug.h \
   direct.h \
   fmm2d.h \
   fsm2d.h \
   ftm2d.h \
   init_band.h \
   iter1d.h \
   iter2d.h \
   iter3d.h \
   mdist.h \
   mDistAux.h \
   param-crv-dst.h \
   msdfSussmanFatemi.h \
   msdfIterBase.h

sources = \
   direct.cpp \
   fmm2d.cpp \
   fsm2d.cpp \
   ftm2d.cpp \
   init_band.cpp \
   iter1d.cpp \
   iter2d.cpp \
   iter3d.cpp \
   param-crv-dst.cpp \
   msdfSussmanFatemi.cpp \
   msdfIterBase.cpp

libtnldistincludedir = $(includedir)/tnldist-0.1

libtnldistinclude_HEADERS = $(headers)


lib_LTLIBRARIES = libtnldist-0.1.la
libtnldist_0_1_la_SOURCES = $(sources) $(headers) debug.h
libtnldist_0_1_la_CXXFLAGS = $(CXXFLAGS) $(OPTIMISECXXFLAGS)

if BUILD_DBG
   lib_LTLIBRARIES += libtnldist-dbg-0.1.la
   libtnldist_dbg_0_1_la_SOURCES = $(sources) $(headers) debug.h
   libtnldist_dbg_0_1_la_CXXFLAGS = $(CXXFLAGS) $(DBGCXXFLAGS)
   libtnldist_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI
   lib_LTLIBRARIES += libtnldist-mpi-0.1.la
   libtnldist_mpi_0_1_la_SOURCES = $(sources) $(headers) debug.h
   libtnldist_mpi_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS)
   libtnldist_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_MPI_DBG
   lib_LTLIBRARIES += libtnldist-mpi-dbg-0.1.la
   libtnldist_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers) debug.h
   libtnldist_mpi_dbg_0_1_la_CXXFLAGS = $(CXXFLAGS) $(MPICXXFLAGS) $(DBGCXXFLAGS)
   libtnldist_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS) $(DBGLDFLAGS)

endif


#bin_PROGRAMS = sdf-restore

#sdf_restore_SOURCES = sdf-restore.cpp \
#                      initial-condition.cpp \
#                      initial-condition.h

#sdf_restore_LDADD = libmdist-0.3.la
#sdf_restore_CXXFLAGS = $(MDIFF_CFLAGS)
#sdf_restore_LDFLAGS = $(MDIFF_LIBS)
