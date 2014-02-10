set( headers tnlArray.h
             tnlArrayOperations.h
             tnlMultiArray.h
             tnlSharedArray.h
             tnlStaticArray.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/arrays )