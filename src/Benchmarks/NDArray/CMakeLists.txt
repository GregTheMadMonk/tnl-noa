add_executable( tnl-benchmark-ndarray tnl-benchmark-ndarray.cpp )
target_compile_options( tnl-benchmark-ndarray PRIVATE ${CXX_TESTS_FLAGS} )
install( TARGETS tnl-benchmark-ndarray RUNTIME DESTINATION bin )

if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-ndarray-cuda tnl-benchmark-ndarray-cuda.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   install( TARGETS tnl-benchmark-ndarray-cuda RUNTIME DESTINATION bin )
endif()
