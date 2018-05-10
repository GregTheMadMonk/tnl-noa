SET( headers BackwardTimeDiscretisation.h
             BoundaryConditionsSetter.h
             ExplicitTimeStepper.h
             ExplicitTimeStepper_impl.h
             ExplicitUpdater.h
             LinearSystemAssembler.h
             MeshDependentTimeSteps.h
             NoTimeDiscretisation.h
             PDESolver.h
             PDESolver_impl.h
             PDESolverTypeResolver.h
             TimeDependentPDESolver.h
             TimeDependentPDESolver_impl.h
             TimeIndependentPDESolver.h
             TimeIndependentPDESolver_impl.h
             SemiImplicitTimeStepper.h
             SemiImplicitTimeStepper_impl.h )

             
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Solvers/PDE )
