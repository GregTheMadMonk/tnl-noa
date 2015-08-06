SET( headers tnlFunctionDiscretizer.h
             tnlFunctionDiscretizer_impl.h
             tnlFunctionEnumerator.h
             tnlFunctionEnumerator_impl.h
             tnlFunctorAdapter.h
             tnlConstantFunction.h
             tnlConstantFunction_impl.h
             tnlExpBumpFunction.h
             tnlExpBumpFunction_impl.h
             tnlSinBumpsFunction.h
             tnlSinBumpsFunction_impl.h
             tnlSinWaveFunction.h
             tnlSinWaveFunction_impl.h
             tnlTestFunction.h
             tnlFunctionType.h
             tnlTestFunction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/functors )
set( common_SOURCES
     ${CURRENT_DIR}/tnlTestFunction_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_functors_CUDA__SOURCES
        ${common_SOURCES} 
        ${CURRENT_DIR}/tnlTestFunction_impl.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_functors_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/functors )