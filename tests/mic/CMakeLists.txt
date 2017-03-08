    ADD_EXECUTABLE( tnlMICArrayTest${mpiExt}${debugExt} ${headers} tnlMICArrayTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMICArrayTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
    TARGET_COMPILE_DEFINITIONS( tnlMICArrayTest${mpiExt}${debugExt} PUBLIC -DHAVE_MIC )

    ADD_EXECUTABLE( tnlMICVectorTest${mpiExt}${debugExt} ${headers} tnlMICVectorTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMICVectorTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
    TARGET_COMPILE_DEFINITIONS( tnlMICVectorTest${mpiExt}${debugExt} PUBLIC -DHAVE_MIC )