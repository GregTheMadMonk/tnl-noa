add_subdirectory( 3rdparty )

install( DIRECTORY TNL/ DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

# TODO: TNL/Experimental/ should be moved somewhere else, the TNL subdirectory
# should contain only header files
add_subdirectory( TNL/Experimental )

# Note that it is important to start building examples as soon as possible,
# because they take the longest time and other stuff can be pipelined before
# they are finished (at least with Ninja).
if( ${WITH_EXAMPLES} )
   add_subdirectory( Examples )
endif()

if( ${WITH_BENCHMARKS} )
   add_subdirectory( Benchmarks )
endif()

if( ${WITH_TOOLS} )
   add_subdirectory( Tools )
endif()

if( ${WITH_TESTS} )
   add_subdirectory( UnitTests )
endif()
if( ${WITH_MATRIX_TESTS} )
   add_subdirectory( UnitTests/Matrices )
endif()

if( ${WITH_PYTHON} )
   add_subdirectory( Python )
endif()
