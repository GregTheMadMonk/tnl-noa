INCLUDE_DIRECTORIES( low-level )

add_subdirectory( low-level )

set (headers tnlArray.h
             tnlAssert.h               
    	     tnlCurve.h 
    	     tnlCudaSupport.h
		     tnlDataElement.h
		     tnlFile.h 
		     tnlFlopsCounter.h 
		     tnlList.h 
		     tnlLongVector.h
		     tnlLongVectorBase.h 
		     tnlLongVectorHost.h 
		     tnlLongVectorCUDA.h  
		     tnlLogger.h 
		     tnlObject.h 
		     tnlStack.h
		     tnlStatistics.h 
		     tnlString.h 
		     tnlReal.h
		     tnlTimerCPU.h  
		     tnlTimerRT.h  
		     tnlVector.h 
			 tnlCudaSupport.h 
		     mfilename.h 
		     mfuncs.h 
		     mpi-supp.h 
		     param-types.h
    )
set( sources tnlFile.cpp
             tnlFlopsCounter.cpp 
             tnlLogger.cpp 
        	 tnlObject.cpp 
        	 tnlStatistics.cpp
        	 tnlString.cpp 
        	 tnlTimerCPU.cpp  
        	 tnlTimerRT.cpp   
        	 mfilename.cpp 
        	 mpi-supp.cpp 
    )

SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlcoreinclude_HEADERS ${headers} )

ADD_LIBRARY( tnlcore${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( tnlcore${debugExt}-${tnlVersion} ${BZIP2_LIBRARIES} )
INSTALL( TARGETS tnlcore${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
    ADD_LIBRARY( tnlcore-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( tnlcore-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}
                                                                ${BZIP2_LIBRARIES}  )
    SET_TARGET_PROPERTIES( tnlcore-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    INSTALL( TARGETS tnlcore-mpi${debugExt}-${tnlVersion} DESTINATION lib )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core )
