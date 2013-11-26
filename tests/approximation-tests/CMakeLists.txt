set( headers tnl-functions-test.h )
             
if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-functions-test${mpiExt}${debugExt} ${headers} tnl-functions-test.cu )
    TARGET_LINK_LIBRARIES( tnl-functions-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
else()
    ADD_EXECUTABLE( tnl-functions-test${mpiExt}${debugExt} ${headers} tnl-functions-test.cpp )
    TARGET_LINK_LIBRARIES( tnl-functions-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
endif()

                                                             
                                                                       
