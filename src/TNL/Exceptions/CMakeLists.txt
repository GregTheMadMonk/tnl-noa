SET( headers CudaBadAlloc.h
             CudaRuntimeError.h
             CudaSupportMissing.h
             MICBadAlloc.h
             MICSupportMissing.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Exceptions )
