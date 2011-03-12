## Process this file with automake to produce Makefile.in

SUBDIRS = core \
          config \
          debug \
          diff \
	       matrix \
	       mesh \
	       solver
	       

lib_LTLIBRARIES = libtnl-0.1.la
libtnl_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_0_1_la_SOURCES = dummy.cpp
libtnl_0_1_la_LIBADD = core/libtnlcore-0.1.la \
                       config/libtnlconfig-0.1.la \
                       diff/libtnldiff-0.1.la \
                       matrix/libtnlmatrix-0.1.la \
                       mesh/libtnlmesh-0.1.la \
                       solver/libtnlsolver-0.1.la \
                       debug/libtnldebug-0.1.la

if BUILD_DBG
lib_LTLIBRARIES += libtnl-dbg-0.1.la
libtnl_dbg_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_dbg_0_1_la_SOURCES = dummy.cpp
libtnl_dbg_0_1_la_LIBADD = core/libtnlcore-dbg-0.1.la \
                           config/libtnlconfig-dbg-0.1.la \
                           diff/libtnldiff-dbg-0.1.la \
                           matrix/libtnlmatrix-dbg-0.1.la \
                           mesh/libtnlmesh-dbg-0.1.la \
                           solver/libtnlsolver-dbg-0.1.la \
                           debug/libtnldebug-dbg-0.1.la
#  debug/libtnldebug-dbg-0.1.la                        
endif                       

if BUILD_MPI
lib_LTLIBRARIES += libtnl-mpi-0.1.la
libtnl_mpi_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_mpi_0_1_la_SOURCES = dummy.cpp
libtnl_mpi_0_1_la_LIBADD = core/libtnlcore-mpi-0.1.la \
                           config/libtnlconfig-mpi-0.1.la \
                           diff/libtnldiff-mpi-0.1.la \
                           matrix/libtnlmatrix-mpi-0.1.la \
                           mesh/libtnlmesh-mpi-0.1.la \
                           solver/libtnlsolver-mpi-0.1.la \
                           debug/libtnldebug-mpi-0.1.la 
endif
                       
if BUILD_MPI_DBG
lib_LTLIBRARIES += libtnl-mpi-dbg-0.1.la
libtnl_mpi_dbg_0_1_la_SOURCES =
# Dummy C++ source to cause C++ linking.
nodist_EXTRA_libtnl_mpi_dbg_0_1_la_SOURCES = dummy.cpp
libtnl_mpi_dbg_0_1_la_LIBADD = core/libtnlcore-mpi-dbg-0.1.la \
                               config/libtnlconfig-mpi-debg-0.1.la \
                               diff/libtnldiff-mpi-dbg-0.1.la \
                               matrix/libtnlmatrix-mpi-dbg-0.1.la \
                               mesh/libtnlmesh-mpi-dbg-0.1.la \
                               solver/libtnlsolver-mpi-dbg-0.1.la \
                               debug/libtnldebug-mpi-dbg-0.1.la
#debug/libtnldebug-mpi-dbg-0.1.la                        
endif                       
