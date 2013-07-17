set( headers tnlCudaDeviceCheckTester.h
             tnlCudaReductionTester.h )
             
if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-device-check-test${mpiExt}${debugExt} ${headers} device-check-test.cu )
    TARGET_LINK_LIBRARIES( tnl-device-check-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

    CUDA_ADD_EXECUTABLE( tnl-reduction-test${mpiExt}${debugExt} ${headers} reduction-test.cu )
    TARGET_LINK_LIBRARIES( tnl-reduction-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                

                                                                                                                                                                                                                                                            
endif()

                                                                       
                                                                       
