SET( EXTRA_DIST tnlConfigDescriptionParser.y 
	            tnlConfigDescriptionScanner.l 	            
    )

SET( headers tnlBasicTypesSetter.h
             tnlConfigDescription.h     	    
 		       tnlConfigDescriptionParser.h 
		       tnlConfigDescriptionParserbase.h 
		       tnlConfigDescriptionParser.ih 
		       tnlConfigDescriptionScanner.h 
		       tnlParameterContainer.h
		       FlexLexer.h 
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/config )
SET( tnl_config_SOURCES 
     ${CURRENT_DIR}/tnlConfigDescription.cpp 
     ${CURRENT_DIR}/tnlConfigDescriptionScanner.cpp 
     ${CURRENT_DIR}/tnlConfigDescriptionParser.cpp 
     ${CURRENT_DIR}/tnlParameterContainer.cpp 
     ${CURRENT_DIR}/parse.cc
    PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/config )
