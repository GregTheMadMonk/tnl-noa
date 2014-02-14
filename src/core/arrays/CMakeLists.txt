set( headers tnlArray.h
             tnlArrayOperations.h
             tnlMultiArray.h
             tnlSharedArray.h
             tnlConstSharedArray.h
             tnlStaticArray.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/arrays )