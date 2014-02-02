set( headers tnlDenseMatrixTester.h
             tnlTridiagonalMatrixTester.h
             tnlMultidiagonalMatrixTester.h
             tnlEllpackMatrixTester.h
             tnlSlicedEllpackMatrixTester.h
             tnlCSRMatrixTester.h )

ADD_EXECUTABLE( tnlDenseMatrixTest${mpiExt}${debugExt} ${headers} tnlDenseMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlDenseMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlDenseMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlDenseMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      

ADD_EXECUTABLE( tnlTridiagonalMatrixTest${mpiExt}${debugExt} ${headers} tnlTridiagonalMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlTridiagonalMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlTridiagonalMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlTridiagonalMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      
                                                                          
ADD_EXECUTABLE( tnlMultidiagonalMatrixTest${mpiExt}${debugExt} ${headers} tnlMultidiagonalMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlMultidiagonalMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlMultidiagonalMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlMultidiagonalMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlMultidiagonalMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()

ADD_EXECUTABLE( tnlEllpackMatrixTest${mpiExt}${debugExt} ${headers} tnlEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlEllpackMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlEllpackMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlEllpackMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlEllpackMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                        
             
ADD_EXECUTABLE( tnlSlicedEllpackMatrixTest${mpiExt}${debugExt} ${headers} tnlSlicedEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlSlicedEllpackMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlSlicedEllpackMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlSlicedEllpackMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlSlicedEllpackMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()

ADD_EXECUTABLE( tnlChunkedEllpackMatrixTest${mpiExt}${debugExt} ${headers} tnlChunkedEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlChunkedEllpackMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlChunkedEllpackMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlChunkedEllpackMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlChunkedEllpackMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()              
        
ADD_EXECUTABLE( tnlCSRMatrixTest${mpiExt}${debugExt} ${headers} tnlCSRMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlCSRMatrixTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlCSRMatrixTest-cuda${mpiExt}${debugExt} ${headers} tnlCSRMatrixTest.cu
                        OPTIONS -arch sm_20 )
   TARGET_LINK_LIBRARIES( tnlCSRMatrixTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt}${debugExt}-0.1 )
endif()             
                                                                                     