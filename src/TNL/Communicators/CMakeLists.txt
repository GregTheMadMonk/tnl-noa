SET( headers MpiCommunicator.h
             NoDistrCommunicator.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Communicators )
