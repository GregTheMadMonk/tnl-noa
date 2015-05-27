SET( headers tnlPDESolver.h
             tnlPDESolver_impl.h
             tnlExplicitTimeStepper.h
             tnlExplicitTimeStepper_impl.h             
             tnlExplicitUpdater.h
             tnlExplicitUpdater_impl.h
             tnlSemiImplicitTimeStepper.h
             tnlSemiImplicitTimeStepper_impl.h
             tnlLinearSystemAssembler.h
             tnlLinearSystemAssembler_impl.h
             tnlBackwardTimeDiscretisation.h
             tnlNoTimeDiscretisation.h  )
             
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/solvers/pde )