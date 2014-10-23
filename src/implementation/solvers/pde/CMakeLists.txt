SET( headers tnlExplicitTimeStepper_impl.h
             tnlPDESolver_impl.h
             tnlExplicitUpdater_impl.h
             tnlLinearSystemAssembler_impl.h  )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/solvers/pde )

