SET( headers Euler.h
             Euler_impl.h
             Merson.h
             Merson_impl.h
             ExplicitSolver.h
             ExplicitSolver_impl.h
             ODESolverMonitor.h                         
             ODESolverMonitor_impl.h )
                
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/ODE )