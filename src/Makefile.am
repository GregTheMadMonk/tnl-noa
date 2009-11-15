## Process this file with automake to produce Makefile.in

SUBDIRS = core \
          debug \
          diff \
          dist \
	      matrix
           
lib_LTLIBRARIES = libtnl-0.1.la
libtnl_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_0_1_la_SOURCES = dummy.cpp
libtnl_0_1_la_LIBADD = debug/libtnldebug-0.1.la \
                       core/libtnlcore-0.1.la \
                       diff/libtnldiff-0.1.la \
                       dist/libtnldist-0.1.la \
                       matrix/libtnlmatrix-0.1.la

if BUILD_DBG
lib_LTLIBRARIES += libtnl-dbg-0.1.la
libtnl_dbg_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_dbg_0_1_la_SOURCES = dummy.cpp
libtnl_dbg_0_1_la_LIBADD = debug/libtnldebug-dbg-0.1.la \
                       core/libtnlcore-dbg-0.1.la \
                       diff/libtnldiff-dbg-0.1.la \
                       dist/libtnldist-dbg-0.1.la \
                       matrix/libtnlmatrix-dbg-0.1.la
endif                       

if BUILD_MPI
lib_LTLIBRARIES += libtnl-mpi-0.1.la
libtnl_mpi_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_mpi_0_1_la_SOURCES = dummy.cpp
libtnl_mpi_0_1_la_LIBADD = debug/libtnldebug-mpi-0.1.la \
                       core/libtnlcore-mpi-0.1.la \
                       diff/libtnldiff-mpi-0.1.la \
                       dist/libtnldist-mpi-0.1.la \
                       matrix/libtnlmatrix-mpi-0.1.la
endif
                       
if BUILD_MPI_DBG
lib_LTLIBRARIES += libtnl-mpi-dbg-0.1.la
libtnl_mpi_dbg_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_mpi_dbg_0_1_la_SOURCES = dummy.cpp
libtnl_mpi_dbg_0_1_la_LIBADD = debug/libtnldebug-mpi-dbg-0.1.la \
                       core/libtnlcore-mpi-dbg-0.1.la \
                       diff/libtnldiff-mpi-dbg-0.1.la \
                       dist/libtnldist-mpi-dbg-0.1.la \
                       matrix/libtnlmatrix-mpi-dbg-0.1.la
endif                       
