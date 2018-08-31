ADD_EXECUTABLE( tnlOperatorFunctionTest${mpiExt} tnlOperatorFunctionTest.cpp )
TARGET_LINK_LIBRARIES( tnlOperatorFunctionTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )
