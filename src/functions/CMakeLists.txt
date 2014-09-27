SET( headers tnlFunctionDiscretizer.h
             tnlConstantFunction.h
             tnlExpBumpFunction.h
             tnlSinBumpsFunction.h
             tnlSinWaveFunction.h
             tnlTestFunction.h )
                    
SET( libtnlfunctionsincludedir ${TNL_INCLUDE_DIR}/functions )
SET( libtnlfunctionsinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/functions )