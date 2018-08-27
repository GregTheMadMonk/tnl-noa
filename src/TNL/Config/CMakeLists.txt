SET( headers 
     ConfigEntryType.h
     ConfigEntryBase.h
     ConfigEntry.h
     ConfigEntryList.h
     ConfigDelimiter.h
     ConfigDescription.h     	    
     ParameterContainer.h
     DefaultBasicTypesChecker.h 
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Config )
set( common_SOURCES
     ${CURRENT_DIR}/ConfigDescription.cpp 
     ${CURRENT_DIR}/ParameterContainer.cpp )
SET( tnl_config_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
SET( tnl_config_CUDA__SOURCES
     ${common_SOURCES} 
     PARENT_SCOPE )
endif()    

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Config )
