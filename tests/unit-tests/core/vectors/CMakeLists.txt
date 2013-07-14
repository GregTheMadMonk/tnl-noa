set( headers tnlVectorTester.h
             tnlSharedVectorTester.h
             tnlCudaVectorOperationsTester.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlCudaVectorOperationsTest${mpiExt}${debugExt} ${headers} tnlCudaVectorOperationsTest.cu )
    TARGET_LINK_LIBRARIES( tnlCudaVectorOperationsTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                
endif()

#ADD_EXECUTABLE( tnlVectorTest${mpiExt}${debugExt} ${headers} tnlVectorTest.cpp )
#TARGET_LINK_LIBRARIES( tnlVectorTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                           tnl${mpiExt}${debugExt}-0.1 )
             


                                                                       