if( BUILD_CUDA )
    cuda_add_executable( tnl-benchmark-blas tnl-benchmark-blas.cu )
    cuda_add_cublas_to_target( tnl-benchmark-blas )
else()
    add_executable( tnl-benchmark-blas tnl-benchmark-blas.cpp )
endif()

find_package( BLAS )
if( BLAS_FOUND )
   target_compile_definitions( tnl-benchmark-blas PUBLIC "-DHAVE_BLAS" )
   target_link_libraries( tnl-benchmark-blas ${BLAS_LIBRARIES} )
else()
   find_library( CBLAS_LIBRARY NAMES cblas
              PATHS /usr/lib
                    /usr/lib64
                    /usr/local/lib
                    /usr/local/lib64 )
   if( CBLAS_LIBRARY )
      target_compile_definitions( tnl-benchmark-blas PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( tnl-benchmark-blas ${CBLAS_LIBRARY} )
   endif()
endif()

install( TARGETS tnl-benchmark-blas RUNTIME DESTINATION bin )
