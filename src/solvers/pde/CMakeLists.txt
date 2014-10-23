SET( headers tnlPDESolver.h
             tnlExplicitTimeStepper.h             
             tnlExplicitUpdater.h
             tnlSemiImplicitTimeStepper.h
             tnlLinearSystemAssembler.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/pde )