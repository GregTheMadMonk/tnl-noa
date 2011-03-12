## Process this file with automake to produce Makefile.in

headers = \
          tnlDistributedGrid.h \
          tnlGrid.h

sources = 
           
if BUILD_CUDA
sources += 
headers +=  
endif

libtnlmeshincludedir = $(TNL_INCLUDE_DIR)/mesh
libtnlmeshinclude_HEADERS = $(headers)

noinst_LTLIBRARIES = libtnlmesh-0.1.la
libtnlmesh_0_1_la_SOURCES = $(sources) $(headers)
libtnlmesh_0_1_la_CXXFLAGS = $(OPTIMISECXXFLAGS)

if BUILD_DBG
   noinst_LTLIBRARIES += libtnlmesh-dbg-0.1.la
   libtnlmesh_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmesh_dbg_0_1_la_CXXFLAGS = $(DBGCXXFLAGS)
   libtnlmesh_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(DBGLDFLAGS)
endif

if BUILD_MPI
   noinst_LTLIBRARIES += libtnlmesh-mpi-0.1.la
   libtnlmesh_mpi_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmesh_mpi_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(OPTIMISECXXFLAGS)
   libtnlmesh_mpi_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS)
endif

if BUILD_MPI_DBG
   noinst_LTLIBRARIES += libtnlmesh-mpi-dbg-0.1.la
   libtnlmesh_mpi_dbg_0_1_la_SOURCES = $(sources) $(headers)
   libtnlmesh_mpi_dbg_0_1_la_CXXFLAGS = $(MPICXXFLAGS) $(DBGCXXFLAGS)
   libtnlmesh_mpi_dbg_0_1_la_LDFLAGS = $(LDFLAGS) $(MPILDFLAGS) $(DBGLDFLAGS)
endif

