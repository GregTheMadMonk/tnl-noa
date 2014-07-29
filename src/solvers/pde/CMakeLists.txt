SET( headers tnlPDESolver.h
             tnlExplicitTimeStepper.h
             tnlExplicitUpdater.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/pde )