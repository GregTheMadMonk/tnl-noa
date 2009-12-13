## Process this file with automake to produce Makefile.in

common = read-file.h

tnlerrnormssources = mdiff-err-norms.cpp \
                     compare-objects.h \
                     compare-objects.cpp \
                     $(common)

tnlgridviewsources = mgrid-view.cpp \
                     $(common)
                     
tnlcurve2gnuplotsources = tnlcurve2gnuplot.cpp \
                          $(common) 

bin_PROGRAMS = tnl-err-norms \
	           tnlgrid-view \
	           tnlcurve2gnuplot

tnl_err_norms_SOURCES = $(tnlerrnormssources)
tnl_err_norms_LDADD = ../../src/libtnl-0.1.la

tnlgrid_view_SOURCES = $(tnlgridviewsources) 
tnlgrid_view_LDADD = ../../src/libtnl-0.1.la

tnlcurve2gnuplot_SOURCES = $(tnlcurve2gnuplotsources)
tnlcurve2gnuplot_LDADD = ../../src/libtnl-0.1.la

if BUILD_DBG
bin_PROGRAMS += tnl-err-norms-dbg \
	            tnlgrid-view-dbg \
	            tnlcurve2gnuplot-dbg

tnl_err_norms_dbg_SOURCES = $(tnlerrnormssources) 
tnl_err_norms_dbg_CXXFLAGS = $(CXXFLAGS) $(DBGCXXFLAGS)
tnl_err_norms_dbg_LDADD = ../../src/libtnl-dbg-0.1.la

tnlgrid_view_dbg_SOURCES = $(tnlgridviewsources) 
tnlgrid_view_dbg_CXXFLAGS = $(CXXFLAGS) $(DBGCXXFLAGS)
tnlgrid_view_dbg_LDADD = ../../src/libtnl-dbg-0.1.la

tnlcurve2gnuplot_dbg_SOURCES = $(tnlcurve2gnuplotsources) 
tnlcurve2gnuplot_dbg_CXXFLAGS = $(CXXFLAGS) $(DBGCXXFLAGS)
tnlcurve2gnuplot_dbg_LDADD = ../../src/libtnl-dbg-0.1.la
endif

bin_SCRIPTS = tnl-diff-err2eoc.py

EXTRA_DIST = debug.h \
	         tnl-diff-err2eoc.py
