ADD_SUBDIRECTORY( Analytic )

SET( headers tnlDomain.h
             tnlExactOperatorFunction.h
             tnlFunctionAdapter.h
             tnlFunctionEvaluator.h
             tnlFunctionEvaluator_impl.h
             tnlMeshFunction.h
             tnlMeshFunction_impl.h
             tnlMeshFunctionEvaluator.h
             tnlMeshFunctionEvaluator_impl.h
             tnlMeshFunctionGnuplotWriter.h
             tnlMeshFunctionGnuplotWriter_impl.h
             tnlMeshFunctionNormGetter.h
             tnlMeshFunctionVTKWriter.h
             tnlMeshFunctionVTKWriter_impl.h             
             tnlOperatorFunction.h
             tnlTestFunction.h             
             tnlTestFunction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Functions )
set( common_SOURCES
     ${CURRENT_DIR}/tnlTestFunction_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_functions_CUDA__SOURCES
        ${common_SOURCES} 
        ${CURRENT_DIR}/tnlTestFunction_impl.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_functions_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Functions )
