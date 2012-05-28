SET( EXTRA_DIST tnlDebugParser.y 
	            tnlDebugScanner.l 
                create-tnl-debug-parser )      

SET(headers tnlDbgDataElement.h 
	        tnlDebugEntry.h 
	        tnlDebugGroup.h 
	        tnlDebugParserbase.h 
	        tnlDebugParser.h 
            tnlDebug.h 
            tnlDebugScanner.h 
            tnlDebugStructure.h )
            
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/debug )
SET( tnl_debug_SOURCES 
#     ${CURRENT_DIR}/tnlDebug.cpp 
#     ${CURRENT_DIR}/tnlDebugGroup.cpp 
#     ${CURRENT_DIR}/tnlDebugStructure.cpp 
#     ${CURRENT_DIR}/tnlDebugParser.cpp 
#     ${CURRENT_DIR}/tnlDebugScanner.cpp 
#     ${CURRENT_DIR}/parse.cc
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/debug )

#ADD_EXECUTABLE( tnl-debug-test tnl-debug-test.cpp )
#TARGET_LINK_LIBRARIES( tnl-debug-test tnldebug${debugExt}-${tnlVersion})

IF( BUILD_MPI )    
    #ADD_EXECUTABLE( tnl-debug-test-mpi tnl-debug-test.cpp )
    #TARGET_LINK_LIBRARIES( tnl-debug-test-mpi tnldebug-mpi${debugExt}-${tnlVersion})
    #SET_TARGET_PROPERTIES( tnl-debug-test-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    
ENDIF()
 