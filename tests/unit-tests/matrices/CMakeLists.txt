set( headers tnlDenseMatrixTester.h )

ADD_EXECUTABLE( tnlDenseMatrixTest${mpiExt}${debugExt} ${headers} tnlDenseMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlDenseMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlDenseMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      
                                                                          