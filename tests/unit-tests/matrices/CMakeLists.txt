set( headers tnlDenseMatrixTester.h
             tnlTridiagonalMatrixTester.h
             tnlMultidiagonalMatrixTester.h
             EllpackMatrixTester.h
             SlicedEllpackMatrixTester.h
             SparseMatrixTester.h )

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

ADD_EXECUTABLE( EllpackMatrixTest${mpiExt} ${headers} EllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( EllpackMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( EllpackMatrixTest-cuda${mpiExt} ${headers} EllpackMatrixTest.cu )
   TARGET_LINK_LIBRARIES( EllpackMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                      tnl )
endif()

ADD_EXECUTABLE( SlicedEllpackMatrixTest${mpiExt} ${headers} SlicedEllpackMatrixTest.cpp )
TARGET_LINK_LIBRARIES( SlicedEllpackMatrixTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl )

if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SlicedEllpackMatrixTest-cuda${mpiExt} ${headers} SlicedEllpackMatrixTest.cu )
   TARGET_LINK_LIBRARIES( SlicedEllpackMatrixTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
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

