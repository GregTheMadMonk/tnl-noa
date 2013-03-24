SET( EXTRA_DIST tnlConfigDescriptionParser.y 
	             tnlConfigDescriptionScanner.l 	            
    )

SET( headers tnlConfigDescription.h     	    
 		       tnlConfigDescriptionParser.h 
		       tnlConfigDescriptionParserbase.h 
		       tnlConfigDescriptionParser.ih 
		       tnlConfigDescriptionScanner.h 
		       tnlParameterContainer.h
		       tnlDefaultBasicTypesChecker.h
		       FlexLexer.h 
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/config )
set( common_SOURCES
     ${CURRENT_DIR}/tnlConfigDescription.cpp 
     ${CURRENT_DIR}/tnlConfigDescriptionScanner.cpp 
     ${CURRENT_DIR}/tnlConfigDescriptionParser.cpp 
     ${CURRENT_DIR}/tnlParameterContainer.cpp 
     ${CURRENT_DIR}/parse.cc )
SET( tnl_config_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
SET( tnl_config_CUDA__SOURCES
     ${common_SOURCES} 
     PARENT_SCOPE )
endif()    

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/config )
