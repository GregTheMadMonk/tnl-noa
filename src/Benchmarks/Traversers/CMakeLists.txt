if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-traversers tnl-benchmark-traversers.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-traversers tnl ${CUDA_cusparse_LIBRARY} )
else()
    ADD_EXECUTABLE( tnl-benchmark-traversers tnl-benchmark-traversers.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-traversers tnl )
endif()

install( TARGETS tnl-benchmark-traversers RUNTIME DESTINATION bin )
