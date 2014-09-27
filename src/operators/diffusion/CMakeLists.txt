SET( headers tnlLinearDiffusion.h
             tnlExactLinearDiffusion.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/diffusion )
