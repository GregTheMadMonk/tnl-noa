set( ENABLE_CODECOVERAGE )

SET( headers matrix-formats-test.h )

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-test-matrix-formats${mpiExt} ${headers} matrix-formats-test.cu )
else()
   ADD_EXECUTABLE( tnl-test-matrix-formats${mpiExt} ${headers} matrix-formats-test.cpp )
endif()

TARGET_LINK_LIBRARIES( tnl-test-matrix-formats${mpiExt} ${CPPUNIT_LIBRARIES}
                                                                   tnl${mpiExt} )


INSTALL( TARGETS tnl-test-matrix-formats
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES tnl-run-matrix-formats-test
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

unset( ENABLE_CODECOVERAGE )
