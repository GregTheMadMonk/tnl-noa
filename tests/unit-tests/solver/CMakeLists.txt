ADD_SUBDIRECTORY( pde )

set( headers tnlSolverTest.h
             tnlSolverTester.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( SolverTest${mpiExt} ${headers} tnlSolverTest.cu )
    TARGET_LINK_LIBRARIES( SolverTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )
else()
    ADD_EXECUTABLE( SolverTest${mpiExt} ${headers} tnlSolverTest.cpp )
    TARGET_LINK_LIBRARIES( SolverTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt} )
endif()

ADD_TEST( SolverTest${mpiExt} ${EXECUTABLE_OUTPUT_PATH}/SolverTest${mpiExt}  ) 
