SET( headers tnlEulerSolver.h
             tnlMersonSolver.h
             tnlExplicitSolver.h
             tnlODESolverMonitor.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/ode )