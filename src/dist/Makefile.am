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
libtnldist_0_1_la_CXXFLAGS = $(MDIFF_CFLAGS)
libtnldist_0_1_la_LDFLAGS = $(MDIFF_LIBS)

#bin_PROGRAMS = sdf-restore

#sdf_restore_SOURCES = sdf-restore.cpp \
#                      initial-condition.cpp \
#                      initial-condition.h

#sdf_restore_LDADD = libmdist-0.3.la
#sdf_restore_CXXFLAGS = $(MDIFF_CFLAGS)
#sdf_restore_LDFLAGS = $(MDIFF_LIBS)
