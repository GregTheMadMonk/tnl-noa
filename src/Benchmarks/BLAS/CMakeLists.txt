if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-blas tnl-benchmark-blas.cu )
    CUDA_ADD_CUBLAS_TO_TARGET( tnl-benchmark-blas )
else()
    ADD_EXECUTABLE( tnl-benchmark-blas tnl-benchmark-blas.cpp )
endif()
target_link_libraries( tnl-benchmark-blas ${BLAS_LIBRARIES} )

install( TARGETS tnl-benchmark-blas RUNTIME DESTINATION bin )
