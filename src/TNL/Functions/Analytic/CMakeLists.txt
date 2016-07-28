SET( headers tnlBlobFunction.h
             tnlBlobFunction_impl.h  
             tnlConstantFunction.h
             tnlConstantFunction_impl.h
             tnlCylinderFunction.h
             tnlCylinderFunction_impl.h  
             tnlExpBumpFunction.h
             tnlExpBumpFunction_impl.h
	     tnlFlowerpotFunction.h
             tnlFlowerpotFunction_impl.h 
	     tnlPseudoSquareFunction.h
             tnlPseudoSquareFunction_impl.h
             tnlSinBumpsFunction.h
             tnlSinBumpsFunction_impl.h
             tnlSinWaveFunction.h
             tnlSinWaveFunction_impl.h
	     tnlTwinsFunction.h
             tnlTwinsFunction_impl.h 
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Functions/Analytic )
