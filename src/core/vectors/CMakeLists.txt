set( headers tnlVector.h
             tnlMultiVector.h
             tnlSharedVector.h
             tnlVectorOperations.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/vectors )