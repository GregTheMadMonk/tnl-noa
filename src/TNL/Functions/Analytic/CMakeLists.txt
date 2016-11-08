SET( headers Blob.h
             Blob_impl.h  
             Constant.h
             Constant_impl.h
             Cylinder.h
             Cylinder_impl.h  
             ExpBump.h
             ExpBump_impl.h
	     Flowerpot.h
             Flowerpot_impl.h 
	     PseudoSquare.h
             PseudoSquare_impl.h
             SinBumps.h
             SinBumps_impl.h
             SinWave.h
             SinWave_impl.h
	     Twins.h
             Twins_impl.h 
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Functions/Analytic )
