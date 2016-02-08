ADD_EXECUTABLE( tnlFDMGradientNormTest${mpiExt}${debugExt} tnlFDMGradientNormTest.cpp )
TARGET_LINK_LIBRARIES( tnlFDMGradientNormTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )

ADD_EXECUTABLE( tnlTwoSidedGradientNormTest${mpiExt}${debugExt} tnlTwoSidedGradientNormTest.cpp )
TARGET_LINK_LIBRARIES( tnlTwoSidedGradientNormTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )

#ADD_EXECUTABLE( tnlCoFVMGradientNormTest${mpiExt}${debugExt} tnlCoFVMGradientNormTest.cpp )
#TARGET_LINK_LIBRARIES( tnlCoFVMGradientNormTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                                 tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlFDMGradientNormTest-cuda${mpiExt}${debugExt} ${headers} tnlFDMGradientNormTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlFDMGradientNormTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )
   
   CUDA_ADD_EXECUTABLE( tnlTwoSidedGradientNormTest-cuda${mpiExt}${debugExt} ${headers} tnlTwoSidedGradientNormTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlTwoSidedGradientNormTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )

   CUDA_ADD_EXECUTABLE( tnlCoFVMGradientNormTest-cuda${mpiExt}${debugExt} ${headers} tnlCoFVMGradientNormTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlCoFVMGradientNormTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )

endif()                                                                      

