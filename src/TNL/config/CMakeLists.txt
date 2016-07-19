SET( headers tnlConfigEntryType.h
             tnlConfigEntryBase.h
             tnlConfigEntry.h
             tnlConfigEntryList.h
             tnlConfigDelimiter.h
             tnlConfigDescription.h     	    
		       tnlParameterContainer.h
		       tnlDefaultBasicTypesChecker.h 
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/config )
set( common_SOURCES
     ${CURRENT_DIR}/tnlConfigDescription.cpp 
     ${CURRENT_DIR}/tnlParameterContainer.cpp )
SET( tnl_config_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
SET( tnl_config_CUDA__SOURCES
     ${common_SOURCES} 
     PARENT_SCOPE )
endif()    

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/config )
