# TODO: Split the benchmark into several files for faster build

#if( BUILD_CUDA )
#    CUDA_ADD_EXECUTABLE( tnl-benchmark-traversers tnl-benchmark-traversers.cu )
#    TARGET_LINK_LIBRARIES( tnl-benchmark-traversers tnl ${CUDA_cusparse_LIBRARY} )
#else()
#    ADD_EXECUTABLE( tnl-benchmark-traversers tnl-benchmark-traversers.cpp )
#    TARGET_LINK_LIBRARIES( tnl-benchmark-traversers tnl )
#endif()
#SET_TARGET_PROPERTIES( tnl-benchmark-traversers PROPERTIES COMPILE_OPTIONS "-g" )

#install( TARGETS tnl-benchmark-traversers RUNTIME DESTINATION bin )
