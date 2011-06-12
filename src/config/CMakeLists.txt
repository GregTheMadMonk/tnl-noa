set( EXTRA_DIST tnlConfigDescriptionParser.y 
	            tnlConfigDescriptionScanner.l 	            
    )

set( headers tnlConfigDescription.h     	    
 		     tnlConfigDescriptionParser.h 
		     tnlConfigDescriptionParserbase.h 
		     tnlConfigDescriptionParser.ih 
		     tnlConfigDescriptionScanner.h 
		     tnlParameterContainer.h
		     FlexLexer.h 
    )
set( sources tnlConfigDescription.cpp 
        	 tnlConfigDescriptionScanner.cpp 
        	 tnlConfigDescriptionParser.cpp 
        	 tnlParameterContainer.cpp 
        	 parse.cc
    )

set( libmconfigincludedir ${TNL_INCLUDE_DIR}/config )
set( libmconfiginclude_HEADERS ${headers} )

add_library( libtnlconfig${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} )

if( BUILD_MPI )
  target_link_libraries( libtnlcore${mpiExt}${debugExt}-0.1 mpi )
endif()

