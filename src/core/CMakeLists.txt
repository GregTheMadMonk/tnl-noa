INCLUDE_DIRECTORIES( low-level )

add_subdirectory( low-level )

set (headers tnlArray.h
             tnlArrayManager.h
             tnlArrayManagerBase.h
             tnlArrayManagerHost.h
             tnlArrayManagerCUDA.h
             tnlAssert.h               
    	       tnlCurve.h 
    	       tnlCudaSupport.h
		       tnlDataElement.h
		       tnlDevice.h
		       tnlFile.h 
		       tnlFlopsCounter.h 
		       tnlList.h 
		       tnlVector.h 
		       tnlVectorHost.h 
		       tnlVectorCUDA.h  
		       tnlLogger.h 
		       tnlObject.h 
		       tnlStack.h
		       tnlStatistics.h 
		       tnlString.h 
		       tnlReal.h
		       tnlTimerCPU.h  
		       tnlTimerRT.h  
		       tnlTuple.h 
			    tnlCudaSupport.h 
		       mfilename.h 
		       mfuncs.h 
		       mpi-supp.h 
		       param-types.h
    )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core )    
set( tnl_core_SOURCES
     ${CURRENT_DIR}/tnlFile.cpp
     ${CURRENT_DIR}/tnlFlopsCounter.cpp 
     ${CURRENT_DIR}/tnlLogger.cpp 
     ${CURRENT_DIR}/tnlObject.cpp 
     ${CURRENT_DIR}/tnlStatistics.cpp
     ${CURRENT_DIR}/tnlString.cpp 
     ${CURRENT_DIR}/tnlTimerCPU.cpp  
     ${CURRENT_DIR}/tnlTimerRT.cpp   
     ${CURRENT_DIR}/mfilename.cpp 
     ${CURRENT_DIR}/mpi-supp.cpp 
    PARENT_SCOPE )

SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlcoreinclude_HEADERS ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core )

#ADD_LIBRARY( tnlcore${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
#TARGET_LINK_LIBRARIES( tnlcore${debugExt}-${tnlVersion} ${BZIP2_LIBRARIES} )
#INSTALL( TARGETS tnlcore${debugExt}-${tnlVersion} DESTINATION lib )

#IF( BUILD_MPI )
#    ADD_LIBRARY( tnlcore-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
#    TARGET_LINK_LIBRARIES( tnlcore-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}
#                                                                ${BZIP2_LIBRARIES}  )
#    SET_TARGET_PROPERTIES( tnlcore-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
#    INSTALL( TARGETS tnlcore-mpi${debugExt}-${tnlVersion} DESTINATION lib )
#ENDIF()


