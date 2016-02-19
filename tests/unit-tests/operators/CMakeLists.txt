ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( fdm )
ADD_SUBDIRECTORY( geometric )

ADD_EXECUTABLE( tnlOperatorCompositionTest${mpiExt}${debugExt} tnlOperatorCompositionTest.cpp )
TARGET_LINK_LIBRARIES( tnlOperatorCompositionTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt}${debugExt}-0.1 )