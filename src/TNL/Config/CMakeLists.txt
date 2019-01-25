SET( headers
     ConfigEntryType.h
     ConfigEntryBase.h
     ConfigEntry.h
     ConfigEntryList.h
     ConfigDelimiter.h
     ConfigDescription.h
     make_unique.h
     ParameterContainer.h
     parseCommandLine.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Config )
