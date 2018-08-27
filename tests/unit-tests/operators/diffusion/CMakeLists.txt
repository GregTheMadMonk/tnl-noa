ADD_EXECUTABLE( tnlLinearDiffusionTest${mpiExt} tnlLinearDiffusionTest.cpp )
TARGET_LINK_LIBRARIES( tnlLinearDiffusionTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )
ADD_EXECUTABLE( tnlOneSidedMeanCurvatureTest${mpiExt} tnlOneSidedMeanCurvatureTest.cpp )
TARGET_LINK_LIBRARIES( tnlOneSidedMeanCurvatureTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                             tnl${mpiExt} )


if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnlLinearDiffusionTest-cuda${mpiExt} ${headers} tnlLinearDiffusionTest.cu )
   TARGET_LINK_LIBRARIES( tnlLinearDiffusionTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt} )
   CUDA_ADD_EXECUTABLE( tnlOneSidedMeanCurvatureTest-cuda${mpiExt} ${headers} tnlOneSidedMeanCurvatureTest.cu )
   TARGET_LINK_LIBRARIES( tnlOneSidedMeanCurvatureTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt} )

endif()                                                                      

