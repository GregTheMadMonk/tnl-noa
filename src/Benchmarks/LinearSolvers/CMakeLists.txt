if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers tnl )
else()
    ADD_EXECUTABLE( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers tnl )
endif()

install( TARGETS tnl-benchmark-linear-solvers RUNTIME DESTINATION bin )
