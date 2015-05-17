SET( headers tnlCylinderFunction.h
             tnlCylinderFunction_impl.h            
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/functions/initial_conditions )
