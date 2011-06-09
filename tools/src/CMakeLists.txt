set(common read-file.h)
set(tnlerrnormssources mdiff-err-norms.cpp 
                     tnl-err-norms.h
                     compare-objects.h
                     ${common}
)

set(tnlgridviewsources tnl-grid-view.cpp
                     tnl-grid-view.h
                     ${common}
)
                     
set(tnlcurve2gnuplotsources tnlcurve2gnuplot.cpp
                          ${common} 
)

set(tnlfunctionsbenchmarksources functions-benchmark.cpp
                               functions-benchmark.h
)

set(tnlmatrixconvertsources tnl-matrix-convert.cpp
                          tnl-matrix-convert.h      
)

if (BUILD_CUDA)
    set(tnlfunctionsbenchmarksources  ${tnlfunctionsbenchmarksources}
			functions-benchmark-cuda.cu 
			functions-benchmark-cuda.cu.h 
    )
    CUDA_ADD_EXECUTABLE(tnl-functions-benchmark ${tnlfunctionsbenchmarksources})
	target_link_libraries (tnl-functions-benchmark libtnl${mpiExt}${debugExt}-0.1)
else (BUILD_CUDA)
    ADD_EXECUTABLE(nl-functions-benchmark ${tnlfunctionsbenchmarksources})
	target_link_libraries (nl-functions-benchmark libtnl${mpiExt}${debugExt}-0.1 )
endif(BUILD_CUDA)

ADD_EXECUTABLE(tnl-err-norms ${tnlerrnormssources})
target_link_libraries (tnl-err-norms libtnl${mpiExt}${debugExt}-0.1 )
ADD_EXECUTABLE(tnlgrid-view ${tnlgridviewsources})
target_link_libraries (tnlgrid-view libtnl${mpiExt}${debugExt}-0.1 )
ADD_EXECUTABLE(tnlcurve2gnuplot ${tnlcurve2gnuplotsources})
target_link_libraries (tnlcurve2gnuplot libtnl${mpiExt}${debugExt}-0.1 )
ADD_EXECUTABLE(tnl-matrix-convert ${tnlmatrixconvertsources})
target_link_libraries (tnl-matrix-convert libtnl${mpiExt}${debugExt}-0.1 )
