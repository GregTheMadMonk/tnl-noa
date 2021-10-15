set( COMMON_EXAMPLES
         ArrayExample
         ArrayExample_forElements
         ArrayViewExample
         ArrayViewExample_forElements
         VectorExample
)

set( MPI_COMMON_EXAMPLES
         DistributedArrayExample
)

SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
   if( ${BUILD_MPI} )
      foreach( target IN ITEMS ${MPI_COMMON_EXAMPLES} )
         cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
         add_custom_command( COMMAND "mpirun" ${mpi_test_parameters} ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
         set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
      endforeach()
   endif()
else()
   foreach( target IN ITEMS ${HOST_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
   endforeach()
   if( ${BUILD_MPI} )
      foreach( target IN ITEMS ${MPI_COMMON_EXAMPLES} )
         add_executable( ${target} ${target}.cpp )
         add_custom_command( COMMAND "mpirun" ${mpi_test_parameters} ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
         set( HOST_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
      endforeach()
   endif()
endif()

IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunContainersExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunContainersExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
