SET( headers Problem.h
             PDEProblem.h
             PDEProblem_impl.h
             HeatEquationProblem.h
             HeatEquationProblem_impl.h
             HeatEquationEocProblem.h
             HeatEquationEocProblem_impl.h             
             HeatEquationEocRhs.h 
	     MeanCurvatureFlowEocProblem.h
	     MeanCurvatureFlowEocProblem_impl.h
	     MeanCurvatureFlowEocRhs.h
	     MeanCurvatureFlowProblem.h
	     MeanCurvatureFlowProblem_impl.h            
   )
   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Problems )
