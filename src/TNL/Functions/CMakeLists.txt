ADD_SUBDIRECTORY( Analytic )

SET( headers Domain.h
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

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Functions )
set( common_SOURCES
     ${CURRENT_DIR}/TestFunction_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_functions_CUDA__SOURCES
        ${common_SOURCES} 
        ${CURRENT_DIR}/TestFunction_impl.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_functions_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Functions )
