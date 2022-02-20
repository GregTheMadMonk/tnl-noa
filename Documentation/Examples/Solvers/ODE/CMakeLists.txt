set( HOST_EXAMPLES
   StaticODESolver-SineExample
   StaticODESolver-LorenzExample
)

set( COMMON_EXAMPLES
   StaticODESolver-SineParallelExample
   StaticODESolver-LorenzParallelExample
   ODESolver-HeatEquationExample
   ODESolver-HeatEquationWithMonitorExample
)

foreach( target IN ITEMS ${HOST_EXAMPLES})
add_executable( ${target} ${target}.cpp )
add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES})
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
   endforeach()
endif()

IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunODESolversExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunODESolversExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()

ADD_CUSTOM_COMMAND( 
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineExample.py 
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.pdf
   OUTPUT StaticODESolver-SineExample.pdf
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
   VERBATIM )

ADD_CUSTOM_TARGET( ProcessODESolversExamplesResults ALL DEPENDS StaticODESolver-SineExample.pdf )