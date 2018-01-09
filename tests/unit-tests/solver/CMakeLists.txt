ADD_SUBDIRECTORY( pde )

set( headers tnlSolverTest.h
             tnlSolverTester.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( SolverTest${mpiExt}${debugExt} ${headers} tnlSolverTest.cu )
    TARGET_LINK_LIBRARIES( SolverTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt} )
else()
    ADD_EXECUTABLE( SolverTest${mpiExt}${debugExt} ${headers} tnlSolverTest.cpp )
    TARGET_LINK_LIBRARIES( SolverTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                              tnl${mpiExt}${debugExt} )
endif()

ADD_TEST( SolverTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/SolverTest${mpiExt}${debugExt}  ) 
