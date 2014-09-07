ADD_EXECUTABLE( tnlLinearDiffusionTest${mpiExt}${debugExt} tnlLinearDiffusionTest.cpp )
TARGET_LINK_LIBRARIES( tnlLinearDiffusionTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlLinearDiffusionTest-cuda${mpiExt}${debugExt} ${headers} tnlLinearDiffusionTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlLinearDiffusionTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      

