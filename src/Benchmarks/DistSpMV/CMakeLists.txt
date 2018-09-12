if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-distributed-spmv-cuda tnl-benchmark-distributed-spmv.cu )
   target_link_libraries( tnl-benchmark-distributed-spmv-cuda tnl )

   install( TARGETS tnl-benchmark-distributed-spmv-cuda RUNTIME DESTINATION bin )
endif()

add_executable( tnl-benchmark-distributed-spmv tnl-benchmark-distributed-spmv.cpp )
target_link_libraries( tnl-benchmark-distributed-spmv tnl )

install( TARGETS tnl-benchmark-distributed-spmv RUNTIME DESTINATION bin )
