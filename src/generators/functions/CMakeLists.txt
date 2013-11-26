SET( headers tnlFunctionDiscretizer.h
             tnlSinWaveFunction.h )
                    
SET( libtnlgeneratorsfunctionsincludedir ${TNL_INCLUDE_DIR}/generators/functions )
SET( libtnlgeneratorsfunctionsinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/generators/functions )