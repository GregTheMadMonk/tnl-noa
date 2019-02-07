SET( headers MpiCommunicator.h
             MpiDefs.h
             MPIPrint.h
             MPITypeResolver.h
             NoDistrCommunicator.h
             ScopedInitializer.h
    )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Communicators )
