if( WITH_PYTHON STREQUAL "yes" )
   ADD_SUBDIRECTORY( Python )
endif( WITH_PYTHON STREQUAL "yes" )

ADD_SUBDIRECTORY( TNL )

if( ${WITH_TOOLS} )
   ADD_SUBDIRECTORY( Tools )
endif()

if( ${WITH_TESTS} )
   ADD_SUBDIRECTORY( UnitTests )
endif()
