SET( EXTRA_DIST tnlConfigDescriptionParser.y 
	            tnlConfigDescriptionScanner.l 	            
    )

SET( headers tnlConfigDescription.h     	    
 		     tnlConfigDescriptionParser.h 
		     tnlConfigDescriptionParserbase.h 
		     tnlConfigDescriptionParser.ih 
		     tnlConfigDescriptionScanner.h 
		     tnlParameterContainer.h
		     FlexLexer.h 
    )
SET( sources tnlConfigDescription.cpp 
        	 tnlConfigDescriptionScanner.cpp 
        	 tnlConfigDescriptionParser.cpp 
        	 tnlParameterContainer.cpp 
        	 parse.cc
    )

SET( libmconfigincludedir ${TNL_INCLUDE_DIR}/config )
SET( libmconfiginclude_HEADERS ${headers} )

ADD_LIBRARY( libtnlconfig${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/config )

if( BUILD_MPI )
    ADD_LIBRARY( libtnlconfig-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
endif()


