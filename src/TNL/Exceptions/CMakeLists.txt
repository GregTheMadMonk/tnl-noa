SET( headers CudaBadAlloc.h
             CudaRuntimeError.h
             CudaSupportMissing.h
             FileDeserializationError.h
             FileSerializationError.h
             MICBadAlloc.h
             MICSupportMissing.h
             MPISupportMissing.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Exceptions )
