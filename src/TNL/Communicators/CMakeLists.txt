SET( headers MpiCommunicator.h
             MpiDefs.h             
             NoDistrCommunicator.h 
    )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Communicators )
