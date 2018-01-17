ADD_EXECUTABLE( tnlFDMGradientNormTest${mpiExt} tnlFDMGradientNormTest.cpp )
TARGET_LINK_LIBRARIES( tnlFDMGradientNormTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )

ADD_EXECUTABLE( tnlTwoSidedGradientNormTest${mpiExt} tnlTwoSidedGradientNormTest.cpp )
TARGET_LINK_LIBRARIES( tnlTwoSidedGradientNormTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )

ADD_EXECUTABLE( tnlCoFVMGradientNormTest${mpiExt} tnlCoFVMGradientNormTest.cpp )
TARGET_LINK_LIBRARIES( tnlCoFVMGradientNormTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlFDMGradientNormTest-cuda${mpiExt} ${headers} tnlFDMGradientNormTest.cu )
   TARGET_LINK_LIBRARIES( tnlFDMGradientNormTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt} )
   
   CUDA_ADD_EXECUTABLE( tnlTwoSidedGradientNormTest-cuda${mpiExt} ${headers} tnlTwoSidedGradientNormTest.cu )
   TARGET_LINK_LIBRARIES( tnlTwoSidedGradientNormTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt} )

   CUDA_ADD_EXECUTABLE( tnlCoFVMGradientNormTest-cuda${mpiExt} ${headers} tnlCoFVMGradientNormTest.cu )
   TARGET_LINK_LIBRARIES( tnlCoFVMGradientNormTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt} )
endif()                                                                      
