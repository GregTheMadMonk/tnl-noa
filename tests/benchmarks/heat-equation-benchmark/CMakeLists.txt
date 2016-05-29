IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation${debugExt} tnl-benchmark-simple-heat-equation.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-simple-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )

    CUDA_ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation-bug${debugExt} tnl-benchmark-simple-heat-equation-bug.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-simple-heat-equation-bug${debugExt} tnl${debugExt}-${tnlVersion} )


    CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation${debugExt} tnl-benchmark-heat-equation.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )
ELSE()
    ADD_EXECUTABLE( tnl-benchmark-heat-equation${debugExt} tnl-benchmark-heat-equation.cpp )    
    TARGET_LINK_LIBRARIES( tnl-benchmark-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )

    ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation${debugExt} tnl-benchmark-simple-heat-equation.cpp )    
    TARGET_LINK_LIBRARIES( tnl-benchmark-simple-heat-equation${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF()


INSTALL( TARGETS 
            tnl-benchmark-heat-equation${debugExt}
            tnl-benchmark-simple-heat-equation${debugExt}
         RUNTIME DESTINATION bin )

if( BUILD_CUDA )
    INSTALL( TARGETS 
                tnl-benchmark-simple-heat-equation-bug${debugExt}
             RUNTIME DESTINATION bin )
endif()


                                            
