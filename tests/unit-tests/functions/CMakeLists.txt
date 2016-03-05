ADD_EXECUTABLE( tnlOperatorFunctionTest${mpiExt}${debugExt} tnlOperatorFunctionTest.cpp )
TARGET_LINK_LIBRARIES( tnlOperatorFunctionTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )
