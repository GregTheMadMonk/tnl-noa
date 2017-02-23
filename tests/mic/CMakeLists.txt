    ADD_EXECUTABLE( tnlSatanExperimentalTest${mpiExt}${debugExt} ${headers} tnlSatanExperimentalTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlSatanExperimentalTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
    #TARGET_COMPILE_DEFINITIONS( tnlSatanExperimentalTest${mpiExt}${debugExt} PUBLIC -DHAVE_MIC )

    ADD_EXECUTABLE( tnlSatanMICVectorExperimentalTest${mpiExt}${debugExt} ${headers} tnlSatanMICVectorExperimentalTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlSatanMICVectorExperimentalTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-0.1 )
    #TARGET_COMPILE_DEFINITIONS( tnlSatanMICVectorExperimentalTest${mpiExt}${debugExt} PUBLIC -DHAVE_MIC )