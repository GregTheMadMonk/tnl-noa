set( headers tnlDenseMatrixTester.h
             tnlTridiagonalMatrixTester.h
             tnlMultidiagonalMatrixTester.h
             tnlEllpackMatrixTester.h
             tnlSlicedEllpackMatrixTester.h
             tnlSparseMatrixTester.h )

ADD_EXECUTABLE( tnlDenseMatrixTest${mpiExt} ${headers} tnlDenseMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlDenseMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlDenseMatrixTest-cuda${mpiExt} ${headers} tnlDenseMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlDenseMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( tnlTridiagonalMatrixTest${mpiExt} ${headers} tnlTridiagonalMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlTridiagonalMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlTridiagonalMatrixTest-cuda${mpiExt} ${headers} tnlTridiagonalMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlTridiagonalMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( tnlMultidiagonalMatrixTest${mpiExt} ${headers} tnlMultidiagonalMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlMultidiagonalMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlMultidiagonalMatrixTest-cuda${mpiExt} ${headers} tnlMultidiagonalMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlMultidiagonalMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( tnlEllpackMatrixTest${mpiExt} ${headers} tnlEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlEllpackMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlEllpackMatrixTest-cuda${mpiExt} ${headers} tnlEllpackMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlEllpackMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( tnlSlicedEllpackMatrixTest${mpiExt} ${headers} tnlSlicedEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlSlicedEllpackMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlSlicedEllpackMatrixTest-cuda${mpiExt} ${headers} tnlSlicedEllpackMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlSlicedEllpackMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( tnlChunkedEllpackMatrixTest${mpiExt} ${headers} tnlChunkedEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlChunkedEllpackMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlChunkedEllpackMatrixTest-cuda${mpiExt} ${headers} tnlChunkedEllpackMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlChunkedEllpackMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( tnlCSRMatrixTest${mpiExt} ${headers} tnlCSRMatrixTest.cpp )
TARGET_LINK_LIBRARIES( tnlCSRMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlCSRMatrixTest-cuda${mpiExt} ${headers} tnlCSRMatrixTest.cu )
   TARGET_LINK_LIBRARIES( tnlCSRMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl${mpiExt} )
endif()

