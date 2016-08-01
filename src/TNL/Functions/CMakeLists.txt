ADD_SUBDIRECTORY( Analytic )

SET( headers Domain.h
             ExactOperatorFunction.h
             FunctionAdapter.h
             MeshFunction.h
             MeshFunction_impl.h
             MeshFunctionEvaluator.h
             MeshFunctionEvaluator_impl.h
             MeshFunctionGnuplotWriter.h
             MeshFunctionGnuplotWriter_impl.h
             MeshFunctionNormGetter.h
             MeshFunctionVTKWriter.h
             MeshFunctionVTKWriter_impl.h             
             OperatorFunction.h
             TestFunction.h             
             TestFunction_impl.h )

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
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Functions )
