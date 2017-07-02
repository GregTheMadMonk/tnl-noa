SET( headers CudaBadAlloc.h
             CudaRuntimeError.h
             CudaSupportMissing.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Exceptions )
