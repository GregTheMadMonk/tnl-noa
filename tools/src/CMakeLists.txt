SET( tnlerrnormssources tnl-err-norms.h
                        tnl-err-norms.cpp
                        compare-objects.h )

SET( tnlgridviewsources tnl-grid-view.cpp
                        tnl-grid-view.h  )
                     
SET( tnlcurve2gnuplotsources tnlcurve2gnuplot.cpp )

SET( tnlfunctionsbenchmarksources functions-benchmark.cpp
                                  functions-benchmark.h )

SET( tnlmatrixconvertsources tnl-matrix-convert.cpp
                             tnl-matrix-convert.h )

ADD_EXECUTABLE( tnl-functions-benchmark ${tnlfunctionsbenchmarksources} )
target_link_libraries( tnl-functions-benchmark tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE(tnl-err-norms ${tnlerrnormssources})
target_link_libraries (tnl-err-norms tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE(tnlgrid-view ${tnlgridviewsources})
target_link_libraries (tnlgrid-view tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE(tnlcurve2gnuplot ${tnlcurve2gnuplotsources})
target_link_libraries (tnlcurve2gnuplot tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE(tnl-matrix-convert ${tnlmatrixconvertsources})
target_link_libraries (tnl-matrix-convert tnl${mpiExt}${debugExt}-0.1 )

INSTALL( TARGETS tnl-err-norms
                 tnlgrid-view
                 tnl-matrix-convert
         RUNTIME DESTINATION bin )