INSTALL( FILES Makefile
               tnlZeroRightHandSide_impl.h
               tnlZeroRightHandSide.h
               tnlTimeFunction_impl.h
               tnlTimeFunction.h
               tnlRightHandSide_impl.h
               tnlRightHandSide.h
               tnlNeumannBoundaryConditions_impl.h
               tnlNeumannBoundaryConditions.h
               tnlLinearDiffusion_impl.h
               tnlLinearDiffusion.h
               tnlDirichletBoundaryConditions_impl.h
               tnlDirichletBoundaryConditions.h
               tnlAnalyticSolution_impl.h
               tnlAnalyticSolution.h
               run-init-mesh
               run-heat-equation-testy-laplace-konvergence
               run-heat-equation-testy-konvergence
               run-heat-equation
               main.cpp
               heatEquationSolver_impl.h
               heatEquationSolver.h
               heatEquationSetter_impl.h
               heatEquationSetter.h
               heat-equation.cfg.desc
               heat-equation-conf
         DESTINATION share/tnl-${tnlVersion}/examples/heat-equation )