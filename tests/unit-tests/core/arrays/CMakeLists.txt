set( headers tnlArrayOperationsTester.h
             tnlArrayTester.h
             tnlMultiArrayTester.h
             tnlSharedArrayTester.h )

ADD_EXECUTABLE( tnlArrayOperationsTest${mpiExt}${debugExt} ${headers} tnlArrayOperationsTest.cpp )
TARGET_LINK_LIBRARIES( tnlArrayOperationsTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE( tnlArrayTest${mpiExt}${debugExt} ${headers} tnlArrayTest.cpp )
TARGET_LINK_LIBRARIES( tnlArrayTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
                                                           
ADD_EXECUTABLE( tnlMultiArrayTest${mpiExt}${debugExt} ${headers} tnlMultiArrayTest.cpp )
TARGET_LINK_LIBRARIES( tnlMultiArrayTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )

             
if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlArrayOperationsTest-cuda${mpiExt}${debugExt} ${headers} tnlArrayOperationsTest.cu 
                         OPTIONS -arch sm_20 -shared )
    TARGET_LINK_LIBRARIES( tnlArrayOperationsTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
                                                              
    CUDA_ADD_EXECUTABLE( tnlArrayTest-cuda${mpiExt}${debugExt} ${headers} tnlArrayTest.cu
                         OPTIONS -arch sm_20 -shared )
    TARGET_LINK_LIBRARIES( tnlArrayTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
                                                              
    CUDA_ADD_EXECUTABLE( tnlMultiArrayTest-cuda${mpiExt}${debugExt} ${headers} tnlMultiArrayTest.cu
                         OPTIONS -arch sm_20 -shared )
    TARGET_LINK_LIBRARIES( tnlMultiArrayTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                
                                                                                                                                                                                                                                                            
endif()
                                                                       