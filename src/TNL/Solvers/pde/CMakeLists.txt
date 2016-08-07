SET( headers tnlBackwardTimeDiscretisation.h
             tnlBoundaryConditionsSetter.h
             tnlBoundaryConditionsSetter_impl.h
             tnlExplicitTimeStepper.h
             tnlExplicitTimeStepper_impl.h
             tnlExplicitUpdater.h
             tnlExplicitUpdater_impl.h
             tnlLinearSystemAssembler.h
             tnlLinearSystemAssembler_impl.h
             tnlNoTimeDiscretisation.h
             tnlPDESolver.h
             tnlPDESolver_impl.h
             tnlSemiImplicitTimeStepper.h
             tnlSemiImplicitTimeStepper_impl.h )

             
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/pde )
