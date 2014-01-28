ADD_SUBDIRECTORY( pde )

set( headers tnlSolverTest.h
             tnlSolverTester.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnlSolverTest${mpiExt}${debugExt} ${headers} tnlSolverTest.cu
                          OPTIONS -arch sm_20 -shared )
    TARGET_LINK_LIBRARIES( tnlSolverTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
else()
    ADD_EXECUTABLE( tnlSolverTest${mpiExt}${debugExt} ${headers} tnlSolverTest.cpp )
    TARGET_LINK_LIBRARIES( tnlSolverTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-0.1 )
endif()

ADD_TEST( tnlSolverTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnlSolverTest${mpiExt}${debugExt}  ) 