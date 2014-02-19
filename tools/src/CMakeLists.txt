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

set( tnl_grid_setup_SOURCES tnl-grid-setup.cpp
                            tnl-grid-setup.h )
                            
set( tnl_mesh_convert_SOURCES tnl-mesh-convert.cpp
                            tnl-mesh-convert.h )
                            
                            
set( tnl_discrete_SOURCES tnl-discrete.cpp
                          tnl-discrete.h )

set( tnl_view_SOURCES tnl-view.cpp
                      tnl-view.h )
                      
set( tnl_diff_SOURCES tnl-diff.cpp
                      tnl-diff.h )                      

ADD_EXECUTABLE(tnl-grid-setup${debugExt} ${tnl_grid_setup_SOURCES})
target_link_libraries (tnl-grid-setup${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-mesh-convert${debugExt} ${tnl_mesh_convert_SOURCES})
target_link_libraries (tnl-mesh-convert${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-discrete${debugExt} ${tnl_discrete_SOURCES})
target_link_libraries (tnl-discrete${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-view${debugExt} ${tnl_view_SOURCES})
target_link_libraries (tnl-view${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-diff${debugExt} ${tnl_diff_SOURCES})
target_link_libraries (tnl-diff${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( tnl-functions-benchmark${debugExt} ${tnlfunctionsbenchmarksources} )
target_link_libraries( tnl-functions-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-err-norms${debugExt} ${tnlerrnormssources})
target_link_libraries (tnl-err-norms${debugExt} tnl${debugExt}-${tnlVersion} )

#ADD_EXECUTABLE(tnl-grid-view${debugExt} ${tnlgridviewsources})
#target_link_libraries (tnl-grid-view${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-curve2gnuplot${debugExt} ${tnlcurve2gnuplotsources})
target_link_libraries (tnl-curve2gnuplot${debugExt} tnl${debugExt}-${tnlVersion} )

#ADD_EXECUTABLE( tnl-matrix-convert${debugExt} ${tnlmatrixconvertsources} )
#target_link_libraries( tnl-matrix-convert${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS tnl-err-norms${debugExt}
#                 tnl-grid-view${debugExt}
                 tnl-discrete${debugExt}
                 tnl-view${debugExt}
                 tnl-diff${debugExt}
                 tnl-grid-setup${debugExt}
                 tnl-mesh-convert${debugExt}
                 tnl-curve2gnuplot${debugExt}
#                 tnl-matrix-convert${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
         
INSTALL( FILES tnl-err2eoc.py
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )         