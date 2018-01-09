ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( fdm )
ADD_SUBDIRECTORY( geometric )

ADD_EXECUTABLE( tnlOperatorCompositionTest${mpiExt} tnlOperatorCompositionTest.cpp )
TARGET_LINK_LIBRARIES( tnlOperatorCompositionTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                 tnl${mpiExt} )
