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
                                                           
             
if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlArrayOperationsTest-cuda${mpiExt}${debugExt} ${headers} tnlArrayOperationsTest.cu )
    TARGET_LINK_LIBRARIES( tnlArrayOperationsTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                
endif()
                                                                       