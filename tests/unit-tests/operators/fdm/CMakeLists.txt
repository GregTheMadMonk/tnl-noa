ADD_EXECUTABLE( tnlFiniteDifferencesTest${mpiExt}${debugExt} tnlFiniteDifferencesTest.cpp )
TARGET_LINK_LIBRARIES( tnlFiniteDifferencesTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlFiniteDifferencesTest-cuda${mpiExt}${debugExt} ${headers} tnlFiniteDifferencesTest.cu
                        OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
   TARGET_LINK_LIBRARIES( tnlFiniteDifferencesTest-cuda${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt}${debugExt}-0.1 )
endif()                                                                      

