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

ADD_EXECUTABLE( tnl-functions-benchmark${debugExt} ${tnlfunctionsbenchmarksources} )
target_link_libraries( tnl-functions-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-err-norms${debugExt} ${tnlerrnormssources})
target_link_libraries (tnl-err-norms${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-grid-view${debugExt} ${tnlgridviewsources})
target_link_libraries (tnl-grid-view${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-curve2gnuplot${debugExt} ${tnlcurve2gnuplotsources})
target_link_libraries (tnl-curve2gnuplot${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( tnl-matrix-convert${debugExt} ${tnlmatrixconvertsources} )
target_link_libraries( tnl-matrix-convert${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS tnl-err-norms${debugExt}
                 tnl-grid-view${debugExt}
                 tnl-curve2gnuplot${debugExt}
                 tnl-matrix-convert${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
         
INSTALL( FILES tnl-err2eoc.py
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )         