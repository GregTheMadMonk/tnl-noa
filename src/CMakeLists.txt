ADD_SUBDIRECTORY( TNL )

if( ${WITH_BENCHMARKS} )
   ADD_SUBDIRECTORY( Benchmarks )
endif()

if( ${WITH_TOOLS} )
   ADD_SUBDIRECTORY( Tools )
endif()

if( ${WITH_TESTS} )
   ADD_SUBDIRECTORY( UnitTests )
endif()

if( ${WITH_PYTHON} )
   ADD_SUBDIRECTORY( Python )
endif()
