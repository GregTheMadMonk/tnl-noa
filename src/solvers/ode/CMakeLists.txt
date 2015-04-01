SET( headers tnlEulerSolver.h
             tnlEulerSolver_impl.h
             tnlMersonSolver.h
             tnlMersonSolver_impl.h
             tnlExplicitSolver.h
             tnlExplicitSolver_impl.h
             tnlODESolverMonitor.h                         
             tnlODESolverMonitor_impl.h )
                
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/ode )