SET( headers CudaBadAlloc.h
             CudaRuntimeError.h
             CudaSupportMissing.h
             MICBadAlloc.h
             MICSupportMissing.h
             MPISupportMissing.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Exceptions )
