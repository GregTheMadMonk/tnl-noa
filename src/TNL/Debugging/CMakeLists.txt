SET( headers
         FPE.h
         MemoryUsage.h
         StackBacktrace.h
)
   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Debugging )
