set( headers tnlVectorTester.h
             tnlSharedVectorTester.h
             tnlVectorOperationsTester.h
             tnlStaticVectorTester.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlVectorOperationsTest-cuda${mpiExt}${debugExt} ${headers} tnlVectorOperationsTest.cu
                         OPTIONS -arch sm_20 )
    TARGET_LINK_LIBRARIES( tnlVectorOperationsTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
    CUDA_ADD_EXECUTABLE( tnlVectorTest-cuda${mpiExt}${debugExt} ${headers} tnlVectorTest.cu
                         OPTIONS -arch sm_20 )
    TARGET_LINK_LIBRARIES( tnlVectorTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                
                                                                                                                              
endif()


ADD_EXECUTABLE( tnlVectorOperationsTest${mpiExt}${debugExt} ${headers} tnlVectorOperationsTest.cpp )
TARGET_LINK_LIBRARIES( tnlVectorOperationsTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                

ADD_EXECUTABLE( tnlVectorTest${mpiExt}${debugExt} ${headers} tnlVectorTest.cpp )
TARGET_LINK_LIBRARIES( tnlVectorTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
             
ADD_EXECUTABLE( tnlStaticVectorTest${mpiExt}${debugExt} ${headers} tnlStaticVectorTest.cpp )
TARGET_LINK_LIBRARIES( tnlStaticVectorTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )


                                                                       