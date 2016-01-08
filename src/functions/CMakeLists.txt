ADD_SUBDIRECTORY( initial_conditions )

SET( headers tnlFunctionEvaluator.h
             tnlFunctionEvaluator_impl.h
             tnlFunctionAdapter.h
             tnlConstantFunction.h
             tnlConstantFunction_impl.h
             tnlExpBumpFunction.h
             tnlExpBumpFunction_impl.h
             tnlSinBumpsFunction.h
             tnlSinBumpsFunction_impl.h
             tnlSinWaveFunction.h
             tnlSinWaveFunction_impl.h
             tnlTestFunction.h
             tnlDomain.h
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
