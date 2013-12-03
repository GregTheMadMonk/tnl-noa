set( headers tnlDenseMatrixTester.h
             tnlTridiagonalMatrixTester.h )

ADD_EXECUTABLE( tnlDenseMatrixTest${mpiExt}${debugExt} ${headers} tnlDenseMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlDenseMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlDenseMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      

ADD_EXECUTABLE( tnlTridiagonalMatrixTest${mpiExt}${debugExt} ${headers} tnlTridiagonalMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlTridiagonalMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlTridiagonalMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      
                                                                          
                                                                          