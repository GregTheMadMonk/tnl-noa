SET( headers tnlFunctionDiscretizer_impl.h
             tnlConstantFunction_impl.h
             tnlExpBumpFunction_impl.h
             tnlSinBumpsFunction_impl.h
             tnlSinWaveFunction_impl.h
             tnlTestFunction_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/functions )
set( common_SOURCES
     ${CURRENT_DIR}/tnlTestFunction_impl.cpp )       

IF( BUILD_CUDA )
   set( tnl_implementation_functions_CUDA__SOURCES
        ${common_SOURCES} 
        ${CURRENT_DIR}/tnlTestFunction_impl.cu
        PARENT_SCOPE )
ENDIF()    

set( tnl_implementation_functions_SOURCES     
     ${common_SOURCES}
     PARENT_SCOPE )
        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/functions )

