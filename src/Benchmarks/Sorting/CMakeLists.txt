if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-sort tnl-benchmark-sort.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-sort ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} )
else()
    ADD_EXECUTABLE( tnl-benchmark-sort tnl-benchmark-sort.cpp )
endif()

install( TARGETS tnl-benchmark-sort RUNTIME DESTINATION bin )
