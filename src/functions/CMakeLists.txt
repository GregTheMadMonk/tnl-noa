ADD_SUBDIRECTORY( initial_conditions )

SET( headers tnlConstantFunction.h
             tnlConstantFunction_impl.h
             tnlDomain.h
             tnlExactOperatorFunction.h
             tnlExpBumpFunction.h
             tnlExpBumpFunction_impl.h
             tnlFunctionAdapter.h
             tnlFunctionEvaluator.h
             tnlFunctionEvaluator_impl.h
             tnlMeshFunction.h
             tnlMeshFunction_impl.h
             tnlMeshFunctionEvaluator.h
             tnlMeshFunctionEvaluator_impl.h
             tnlMeshFunctionGnuplotWriter.h
             tnlMeshFunctionGnuplotWriter_impl.h
             tnlMeshFunctionVTKWriter.h
             tnlMeshFunctionVTKWriter_impl.h
             tnlMeshFunctionNormGetter.h
             tnlOperatorFunction.h
             tnlSinBumpsFunction.h
             tnlSinBumpsFunction_impl.h
             tnlSinWaveFunction.h
             tnlSinWaveFunction_impl.h
             tnlTestFunction.h             
             tnlTestFunction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/functions )
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
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/functions )
