ADD_SUBDIRECTORY( level_set_functions )

SET( headers tnlCylinderFunction.h
             tnlCylinderFunction_impl.h  
	     tnlFlowerpotFunction.h
             tnlFlowerpotFunction_impl.h 
	     tnlTwinsFunction.h
             tnlTwinsFunction_impl.h             
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/functors/initial_conditions )
