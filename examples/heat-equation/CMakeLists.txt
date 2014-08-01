set( tnl_heat_equation_SOURCES
               tnlZeroRightHandSide_impl.h
               tnlZeroRightHandSide.h
               tnlTimeFunction_impl.h
               tnlTimeFunction.h
               tnlRightHandSide_impl.h
               tnlRightHandSide.h
               tnlNeumannBoundaryConditions_impl.h
               tnlNeumannBoundaryConditions.h
               tnlDirichletBoundaryConditions_impl.h
               tnlDirichletBoundaryConditions.h
               tnlAnalyticSolution_impl.h
               tnlAnalyticSolution.h
               main.cpp
               heatEquationSolver_impl.h
               heatEquationSolver.h
               heatEquationSetter_impl.h
               heatEquationSetter.h 
               heatEquationConfig.h )
               
ADD_EXECUTABLE(tnl-heat-equation${debugExt} ${tnl_heat_equation_SOURCES})
target_link_libraries (tnl-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS tnl-heat-equation${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES #Makefile
               run-init-mesh
               run-heat-equation-testy-laplace-konvergence
               run-heat-equation-testy-konvergence
               run-heat-equation
               heat-equation.cfg.desc
					debugging_parameters
					EOCtest
					EOCtest-adaptivity-cycle

         DESTINATION share/tnl-${tnlVersion}/examples/heat-equation )
