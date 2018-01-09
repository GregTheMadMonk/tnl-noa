ADD_SUBDIRECTORY( share )
ADD_SUBDIRECTORY( heat-equation-benchmark )

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-blas tnl-benchmark-blas.cu )
    if( HAVE_CUBLAS STREQUAL "yes" )
        CUDA_ADD_CUBLAS_TO_TARGET( tnl-benchmark-blas )
    endif()
    TARGET_LINK_LIBRARIES( tnl-benchmark-blas tnl ${CUSPARSE_LIBRARY} )

    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv tnl ${CUSPARSE_LIBRARY} )

    CUDA_ADD_EXECUTABLE( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers tnl ${CUSPARSE_LIBRARY} )
ELSE()
    ADD_EXECUTABLE( tnl-benchmark-blas tnl-benchmark-blas.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-blas tnl )

    ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv tnl )

    ADD_EXECUTABLE( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers tnl )
ENDIF()

INSTALL( TARGETS
            tnl-benchmark-blas
            tnl-benchmark-spmv
            tnl-benchmark-linear-solvers
         RUNTIME DESTINATION bin )
