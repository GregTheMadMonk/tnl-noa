ADD_SUBDIRECTORY( cuda )

set( headers tnlFileTester.h
             tnlStringTester.h
             tnlObjectTester.h
             tnlRealTester.h
             tnlTupleTester.h
             tnlVectorTester.h
             tnlMultiArrayTester.h
             tnlArrayTester.h
             tnlSharedArrayTester.h
             tnlSharedVectorTester.h
             tnlGridOldTester.h
             tnlSharedMemoryTester.h
             tnlCommunicatorTester.h )
             
if( BUILD_CUDA )
#    CUDA_ADD_EXECUTABLE( tnl-device-check-test${mpiExt}${debugExt} ${headers} device-check-test.cu )
#    TARGET_LINK_LIBRARIES( tnl-device-check-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt}${debugExt}-0.1 )
#
#    CUDA_ADD_EXECUTABLE( tnl-memory-operations-test${mpiExt}${debugExt} ${headers} memory-operations-test.cu )
#    TARGET_LINK_LIBRARIES( tnl-memory-operations-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt}${debugExt}-0.1 )
#    
#    CUDA_ADD_EXECUTABLE( tnl-reduction-test${mpiExt}${debugExt} ${headers} reduction-test.cu )
#    TARGET_LINK_LIBRARIES( tnl-reduction-test${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
#                                                              tnl${mpiExt}${debugExt}-0.1 )                                                                
#                                                                                                                                                                                                                                                            
endif()
                                                                       