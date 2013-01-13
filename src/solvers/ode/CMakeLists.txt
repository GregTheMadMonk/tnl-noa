SET( headers tnlEulerSolver.h
             tnlMersonSolver.h
             tnlExplicitSolver.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/ode )