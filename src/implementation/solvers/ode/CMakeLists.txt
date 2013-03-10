SET( headers tnlEulerSolver_impl.h
             tnlMersonSolver_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/ode )

