## Process this file witj automake to produce Makefile.in

SUBDIRS = src

pkgconfigdir = $(libdir)/pkgconfig
if BUILDMPI
pkgconfig_DATA = libtnl-0.1.pc libtnl-mpi-0.1.pc
else
pkgconfig_DATA = libtnl-0.1.pc
endif

EXTRA_DIST = TODO create-config-description-parser
