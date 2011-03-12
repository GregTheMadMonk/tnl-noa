## Process this file witj automake to produce Makefile.in

SUBDIRS = src \
          tools \
          tests

          
libtnlincludedir = $(TNL_INCLUDE_DIR)
libtnlinclude_HEADERS = tnl-config.h

pkgconfigdir = $(libdir)/pkgconfig

pkgconfig_DATA = libtnl-$(VERSION).pc

if BUILD_MPI_DBG
  pkgconfig_DATA += libtnl-mpi-dbg-$(VERSION).pc
endif

if BUILD_MPI
  pkgconfig_DATA += libtnl-mpi-$(VERSION).pc
endif

if BUILD_DBG
   pkgconfig_DATA += libtnl-dbg-$(VERSION).pc
endif

EXTRA_DIST = TODO 
