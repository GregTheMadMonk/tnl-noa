SET( headers tnlProblem.h
             tnlPDEProblem.h
             tnlPDEProblem_impl.h
             tnlHeatEquationProblem.h
             tnlHeatEquationProblem_impl.h
             tnlHeatEquationEocProblem.h
             tnlHeatEquationEocProblem_impl.h             
             tnlHeatEquationEocRhs.h             
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/problems )
