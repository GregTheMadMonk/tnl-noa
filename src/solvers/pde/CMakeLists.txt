SET( headers tnlPDESolver.h
             tnlExplicitTimeStepper.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/pde )