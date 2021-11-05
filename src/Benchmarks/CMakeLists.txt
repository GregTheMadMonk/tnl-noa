add_subdirectory( HeatEquation )
add_subdirectory( BLAS )
add_subdirectory( NDArray )
add_subdirectory( SpMV )
add_subdirectory( DistSpMV )
add_subdirectory( LinearSolvers )
add_subdirectory( ODESolvers )
add_subdirectory( Sorting )
add_subdirectory( Traversers )

set( headers
         Benchmarks.h
         Benchmarks.hpp
         FunctionTimer.h
         Logging.h
         CustomLogging.h
         JsonLogging.h
)

install( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Benchmarks )
