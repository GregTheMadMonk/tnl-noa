add_subdirectory( HeatEquation )
add_subdirectory( BLAS )
add_subdirectory( SpMV )
add_subdirectory( DistSpMV )
add_subdirectory( LinearSolvers )
add_subdirectory( Traversers )

set( headers
         Benchmarks.h
)

install( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Benchmarks )
