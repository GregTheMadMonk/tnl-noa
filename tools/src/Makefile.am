## Process this file with automake to produce Makefile.in

common =

tnlerrnormssources = tnl-err-norms.h \
                     compare-objects.h \
                     $(common)
if BUILD_CUDA
tnlerrnormssources +=  tnl-err-norms-cuda.cu
else
tnlerrnormssources +=  tnl-err-norms.cpp
endif

tnlgridviewsources = tnl-grid-view.h \
                     $(common)
                     
#if BUILD_CUDA
#tnlgridviewsources += tnl-grid-view-cuda.cu
#else
tnlgridviewsources += tnl-grid-view.cpp
#endif                     
                     
tnlcurve2gnuplotsources = tnlcurve2gnuplot.cpp \
                          $(common) 
                                                    
tnlfunctionsbenchmarksources = functions-benchmark.cpp \
                               functions-benchmark.h

tnlmatrixconvertsources = tnl-matrix-convert.cpp \
                          tnl-matrix-convert.h                                

#if BUILD_CUDA
#tnlfunctionsbenchmarksources += functions-benchmark-cuda.cu \
#                                functions-benchmark-cuda.cu.h                                 
#endif

bin_PROGRAMS = tnl-err-norms \
	            tnlgrid-view \
	            tnlcurve2gnuplot \
	            tnl-functions-benchmark \
	            tnl-matrix-convert

tnl_err_norms_SOURCES = $(tnlerrnormssources)
tnl_err_norms_LDADD = ../../src/libtnl-0.1.la

tnlgrid_view_SOURCES = $(tnlgridviewsources) 
tnlgrid_view_LDADD = ../../src/libtnl-0.1.la

tnlcurve2gnuplot_SOURCES = $(tnlcurve2gnuplotsources)
tnlcurve2gnuplot_LDADD = ../../src/libtnl-0.1.la

tnl_functions_benchmark_SOURCES = $(tnlfunctionsbenchmarksources)
tnl_functions_benchmark_LDADD = ../../src/libtnl-0.1.la
tnl_functions_benchmark_CXXFLAGS = "-O2" 

tnl_matrix_convert_SOURCES = $(tnlmatrixconvertsources)
tnl_matrix_convert_CXXFLAGS = $(OPTIMISECXXFLAGS)
tnl_matrix_convert_LDADD = ../../src/libtnl-0.1.la

if BUILD_DBG
bin_PROGRAMS += tnl-err-norms-dbg \
	             tnlgrid-view-dbg \
	             tnlcurve2gnuplot-dbg \
	             tnl-matrix-convert-dbg

tnl_err_norms_dbg_SOURCES = $(tnlerrnormssources) 
tnl_err_norms_dbg_CXXFLAGS = $(DBGCXXFLAGS)
tnl_err_norms_dbg_LDADD = ../../src/libtnl-dbg-0.1.la

tnlgrid_view_dbg_SOURCES = $(tnlgridviewsources) 
tnlgrid_view_dbg_CXXFLAGS = $(DBGCXXFLAGS)
tnlgrid_view_dbg_LDADD = ../../src/libtnl-dbg-0.1.la

tnlcurve2gnuplot_dbg_SOURCES = $(tnlcurve2gnuplotsources) 
tnlcurve2gnuplot_dbg_CXXFLAGS = $(DBGCXXFLAGS)
tnlcurve2gnuplot_dbg_LDADD = ../../src/libtnl-dbg-0.1.la

tnl_matrix_convert_dbg_SOURCES = $(tnlmatrixconvertsources)
tnl_matrix_convert_dbg_CXXFLAGS = $(DBGCXXFLAGS)
tnl_matrix_convert_dbg_LDADD = ../../src/libtnl-0.1.la
endif

bin_SCRIPTS = tnl-diff-err2eoc.py \
              tnl-create-debug-file

EXTRA_DIST = tnl-diff-err2eoc.py \
             tnl-create-debug-file
