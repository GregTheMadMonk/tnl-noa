ADD_EXECUTABLE( tnlLinearDiffusionTest${mpiExt}${debugExt} tnlLinearDiffusionTest.cpp )
TARGET_LINK_LIBRARIES( tnlLinearDiffusionTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )
ADD_EXECUTABLE( tnlOneSidedTotalVariationMinimizationTest${mpiExt}${debugExt} tnlOneSidedTotalVariationMinimizationTest.cpp )
TARGET_LINK_LIBRARIES( tnlOneSidedTotalVariationMinimizationTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                             tnl${mpiExt}${debugExt}-0.1 )


if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlLinearDiffusionTest-cuda${mpiExt}${debugExt} ${headers} tnlLinearDiffusionTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlLinearDiffusionTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )
   CUDA_ADD_EXECUTABLE( tnlOneSidedNonlinearDiffusionTest-cuda${mpiExt}${debugExt} ${headers} tnlOneSidedNonlinearDiffusionTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlOneSidedNonlinearDiffusionTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )

endif()                                                                      

