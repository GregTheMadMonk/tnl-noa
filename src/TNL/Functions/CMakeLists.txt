ADD_SUBDIRECTORY( Analytic )

SET( headers BoundaryMeshFunction.h
             Domain.h
             ExactOperatorFunction.h
             FunctionAdapter.h
             MeshFunction.h
             MeshFunction_impl.h
             MeshFunctionEvaluator.h
             MeshFunctionEvaluator_impl.h
             MeshFunctionGnuplotWriter.h
             MeshFunctionNormGetter.h
             MeshFunctionVTKWriter.h
             OperatorFunction.h
             TestFunction.h
             TestFunction_impl.h
             VectorField.h
             VectorFieldEvaluator.h
             VectorFieldEvaluator_impl.h
             VectorFieldGnuplotWriter.h
             VectorFieldGnuplotWriter_impl.h
             VectorFieldVTKWriter.h
             VectorFieldVTKWriter_impl.h
             CutMeshFunction.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Functions )
