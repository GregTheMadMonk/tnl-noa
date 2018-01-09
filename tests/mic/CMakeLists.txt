    ADD_EXECUTABLE( tnlMICArrayTest${mpiExt} ${headers} tnlMICArrayTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMICArrayTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt} )
    TARGET_COMPILE_DEFINITIONS( tnlMICArrayTest${mpiExt} PUBLIC ${MIC_CXX_FLAGS} )

    ADD_EXECUTABLE( tnlMICVectorTest${mpiExt} ${headers} tnlMICVectorTest.cpp )   
    TARGET_LINK_LIBRARIES( tnlMICVectorTest${mpiExt} ${CPPUNIT_LIBRARIES}
                                                           tnl${mpiExt} )
    TARGET_COMPILE_DEFINITIONS( tnlMICVectorTest${mpiExt} PUBLIC ${MIC_CXX_FLAGS} )
