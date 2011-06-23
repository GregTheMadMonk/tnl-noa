SET( EXTRA_DIST tnlDebugParser.y 
	            tnlDebugScanner.l 
                create-tnl-debug-parser )      

SET( sources tnlDebug.cpp 
             tnlDebugGroup.cpp 
             tnlDebugStructure.cpp 
	         tnlDebugParser.cpp 
	         tnlDebugScanner.cpp 
	         parse.cc )
	         
SET(headers tnlDbgDataElement.h 
	        tnlDebugEntry.h 
	        tnlDebugGroup.h 
	        tnlDebugParserbase.h 
	        tnlDebugParser.h 
            tnlDebug.h 
            tnlDebugScanner.h 
            tnlDebugStructure.h )

ADD_LIBRARY( tnldebug${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
ADD_EXECUTABLE( tnl-debug-test tnl-debug-test.cpp )
TARGET_LINK_LIBRARIES( tnl-debug-test tnldebug${debugExt}-${tnlVersion})
INSTALL( TARGETS tnldebug${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
    ADD_LIBRARY( tnldebug-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    SET_TARGET_PROPERTIES( tnldebug-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    
    ADD_EXECUTABLE( tnl-debug-test-mpi tnl-debug-test.cpp )
    TARGET_LINK_LIBRARIES( tnl-debug-test-mpi tnldebug-mpi${debugExt}-${tnlVersion})
    SET_TARGET_PROPERTIES( tnl-debug-test-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    
    INSTALL( TARGETS tnldebug-mpi${debugExt}-${tnlVersion} DESTINATION lib )
ENDIF()


INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/debug ) 