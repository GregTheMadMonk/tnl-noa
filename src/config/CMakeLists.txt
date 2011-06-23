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

ADD_LIBRARY( tnlconfig${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/config )
INSTALL( TARGETS tnlconfig${debugExt}-${tnlVersion} DESTINATION lib )

if( BUILD_MPI )
    ADD_LIBRARY( tnlconfig-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    SET_TARGET_PROPERTIES( tnlconfig-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    INSTALL( TARGETS tnlconfig-mpi${debugExt}-${tnlVersion} DESTINATION lib )
endif()


