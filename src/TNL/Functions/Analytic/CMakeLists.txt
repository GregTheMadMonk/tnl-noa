SET( headers BlobFunction.h
             BlobFunction_impl.h  
             ConstantFunction.h
             ConstantFunction_impl.h
             CylinderFunction.h
             CylinderFunction_impl.h  
             ExpBumpFunction.h
             ExpBumpFunction_impl.h
	     FlowerpotFunction.h
             FlowerpotFunction_impl.h 
	     PseudoSquareFunction.h
             PseudoSquareFunction_impl.h
             SinBumpsFunction.h
             SinBumpsFunction_impl.h
             SinWaveFunction.h
             SinWaveFunction_impl.h
	     TwinsFunction.h
             TwinsFunction_impl.h 
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Functions/Analytic )
