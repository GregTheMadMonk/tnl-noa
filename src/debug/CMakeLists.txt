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

ADD_LIBRARY( libtnldebug${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
ADD_EXECUTABLE( tnl-debug-test tnl-debug-test.cpp )
TARGET_LINK_LIBRARIES( tnl-debug-test libtnldebug${debugExt}-${tnlVersion})

IF( BUILD_MPI )
    ADD_LIBRARY( libtnldebug-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    ADD_EXECUTABLE( tnl-debug-test-mpi tnl-debug-test.cpp )
    TARGET_LINK_LIBRARIES( tnl-debug-test-mpi libtnldebug-mpi${debugExt}-${tnlVersion})
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
ENDIF()


INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/debug ) 