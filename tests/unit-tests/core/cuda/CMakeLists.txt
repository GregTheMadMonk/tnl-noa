set( headers tnlCudaTester.h
             tnlCudaReductionTester.h )
             
if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlCudaTest${mpiExt}${debugExt} ${headers} tnlCudaTest.cu
                         OPTIONS -arch sm_20 )
    TARGET_LINK_LIBRARIES( tnlCudaTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

    CUDA_ADD_EXECUTABLE( tnl-reduction-test${mpiExt}${debugExt} ${headers} reduction-test.cu
                         OPTIONS -arch sm_20 )
    TARGET_LINK_LIBRARIES( tnl-reduction-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                

                                                                                                                                                                                                                                                            
endif()

                                                                       
                                                                       
