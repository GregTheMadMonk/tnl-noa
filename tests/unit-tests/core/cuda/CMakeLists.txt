set( headers tnlCudaTester.h
             tnlCudaReductionTester.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlCudaTest${mpiExt} ${headers} tnlCudaTest.cu )
    TARGET_LINK_LIBRARIES( tnlCudaTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )

    CUDA_ADD_EXECUTABLE( tnl-reduction-test${mpiExt} ${headers} reduction-test.cu )
    TARGET_LINK_LIBRARIES( tnl-reduction-test${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )


endif()

