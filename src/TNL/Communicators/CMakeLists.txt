SET( headers MpiCommunicator.h
             NoDistrCommunicator.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Communicators )
