SET( headers tnlPDESolver.h
             tnlExplicitTimeStepper.h             
             tnlExplicitUpdater.h
             tnlSemiImplicitTimeStepper.h
             tnlLinearSystemInitiator.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/pde )