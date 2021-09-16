set( CPP_TESTS MPIUtilsTest )

if( ${BUILD_MPI} )
foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${GTEST_BOTH_LIBRARIES} )

   foreach( np IN ITEMS 2 3 4 )
      set( mpirun_parameters -np ${np} -H localhost:${np} )
      add_test( NAME ${target}_np${np} COMMAND mpirun ${mpirun_parameters} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
   endforeach()
endforeach()
endif()
