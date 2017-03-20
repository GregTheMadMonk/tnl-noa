SET( headers BackwardTimeDiscretisation.h
             BoundaryConditionsSetter.h
             ExplicitTimeStepper.h
             ExplicitTimeStepper_impl.h
             ExplicitUpdater.h
             ExplicitUpdater_impl.h
             LinearSystemAssembler.h
             LinearSystemAssembler_impl.h
             NoTimeDiscretisation.h
             TimeDependentPDESolver.h
             TimeDependentPDESolver_impl.h
             TimeIndependentPDESolver.h
             TimeIndependentPDESolver_impl.h
             SemiImplicitTimeStepper.h
             SemiImplicitTimeStepper_impl.h )

             
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Solvers/PDE )
