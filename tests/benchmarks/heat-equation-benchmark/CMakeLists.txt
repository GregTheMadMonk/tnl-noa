IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation${debugExt} tnl-benchmark-heat-equation.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )

ELSE()
    ADD_EXECUTABLE( tnl-benchmark-heat-equation${debugExt} tnl-benchmark-heat-equation.cpp )    
    TARGET_LINK_LIBRARIES( tnl-benchmark-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF()


INSTALL( TARGETS tnl-benchmark-heat-equation${debugExt}
         RUNTIME DESTINATION bin )


                                            
