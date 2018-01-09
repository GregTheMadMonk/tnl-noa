ADD_EXECUTABLE( tnlFiniteDifferencesTest${mpiExt} tnlFiniteDifferencesTest.cpp )
TARGET_LINK_LIBRARIES( tnlFiniteDifferencesTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )

if( BUILD_CUDA )                                                           
   CUDA_ADD_EXECUTABLE( tnlFiniteDifferencesTest-cuda${mpiExt} ${headers} tnlFiniteDifferencesTest.cu )
   TARGET_LINK_LIBRARIES( tnlFiniteDifferencesTest-cuda${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                          tnl${mpiExt} )
endif()                                                                      

