ADD_EXECUTABLE( tnlFDMGradientNormTest${mpiExt}${debugExt} tnlFDMGradientNormTest.cpp )
TARGET_LINK_LIBRARIES( tnlFDMGradientNormTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlFDMGradientNormTest-cuda${mpiExt}${debugExt} ${headers} tnlFDMGradientNormTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlFDMGradientNormTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      

