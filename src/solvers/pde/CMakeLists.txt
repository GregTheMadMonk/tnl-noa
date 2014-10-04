SET( headers tnlPDESolver.h
             tnlExplicitTimeStepper.h
             tnlSemiImplicitTimeStepper.h
             tnlExplicitUpdater.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/pde )