IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation tnl-benchmark-simple-heat-equation.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-simple-heat-equation tnl )

    CUDA_ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation-bug tnl-benchmark-simple-heat-equation-bug.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-simple-heat-equation-bug tnl )


    CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-heat-equation tnl )
ELSE()
    ADD_EXECUTABLE( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )    
    TARGET_LINK_LIBRARIES( tnl-benchmark-heat-equation tnl )

    ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation tnl-benchmark-simple-heat-equation.cpp )    
    TARGET_LINK_LIBRARIES( tnl-benchmark-simple-heat-equation tnl )
ENDIF()


INSTALL( TARGETS 
            tnl-benchmark-heat-equation
            tnl-benchmark-simple-heat-equation
         RUNTIME DESTINATION bin )

if( BUILD_CUDA )
    INSTALL( TARGETS 
                tnl-benchmark-simple-heat-equation-bug
             RUNTIME DESTINATION bin )
endif()


                                            
