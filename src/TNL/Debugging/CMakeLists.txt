SET( headers
         FPE.h
         MemoryUsage.h
         StackBacktrace.h
)
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Debugging )
