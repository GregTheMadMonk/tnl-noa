set( tnl_heat_equation_SOURCES
     tnlHeatEquationEocRhs.h
     tnlRightHandSide_impl.h
     tnlRightHandSide.h
     tnl-heat-equation.cpp
     tnl-heat-equation-eoc.cpp
     heatEquationSolver_impl.h
     heatEquationSolver.h
     tnlHeatEquationEocRhs.h )
               
ADD_EXECUTABLE(tnl-heat-equation${debugExt} tnl-heat-equation.cpp)
target_link_libraries (tnl-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-heat-equation-eoc-test${debugExt} tnl-heat-equation-eoc.cpp)
target_link_libraries (tnl-heat-equation-eoc-test${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS tnl-heat-equation${debugExt}
                 tnl-heat-equation-eoc-test${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES ${tnl_heat_equation_SOURCES}
               Makefile
               tnl-run-heat-equation-eoc-test
               tnl-run-heat-equation
         DESTINATION share/tnl-${tnlVersion}/examples/heat-equation )
