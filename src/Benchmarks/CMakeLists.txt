add_subdirectory( HeatEquation )
add_subdirectory( BLAS )
add_subdirectory( SpMV )
add_subdirectory( LinearSolvers )

set( headers
         Benchmarks.h
)

install( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Benchmarks )
