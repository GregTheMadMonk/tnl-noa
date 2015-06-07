SET( headers tnlBlobFunction.h
             tnlBlobFunction_impl.h  
	     tnlPseudoSquareFunction.h
             tnlPseudoSquareFunction_impl.h          
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/functors/initial_conditions/level_set_functions )
