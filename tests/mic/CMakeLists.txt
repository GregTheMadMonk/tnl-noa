    ADD_EXECUTABLE( tnlMICArrayTest${mpiExt}${debugExt} ${headers} tnlMICArrayTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMICArrayTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt} )
    TARGET_COMPILE_DEFINITIONS( tnlMICArrayTest${mpiExt}${debugExt} PUBLIC ${MIC_CXX_FLAGS} )

    ADD_EXECUTABLE( tnlMICVectorTest${mpiExt}${debugExt} ${headers} tnlMICVectorTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMICVectorTest${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt}${debugExt} )
    TARGET_COMPILE_DEFINITIONS( tnlMICVectorTest${mpiExt}${debugExt} PUBLIC ${MIC_CXX_FLAGS} )
