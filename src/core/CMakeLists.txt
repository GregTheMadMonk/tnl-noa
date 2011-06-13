INCLUDE_DIRECTORIES( low-level )

add_subdirectory( low-level )

set (headers tnlAssert.h  
    	     tnlCurve.h 
    	     tnlCudaSupport.h
		     tnlDataElement.h
		     tnlFile.h 
		     tnlFlopsCounter.h 
		     tnlList.h 
		     tnlLongVectorBase.h 
		     tnlLongVectorHost.h 
		     tnlLongVectorCUDA.h  
		     tnlLogger.h 
		     tnlObject.h 
		     tnlStack.h 
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
set( sources tnlCudaSupport.cpp 
             tnlFile.cpp
             tnlFlopsCounter.cpp 
             tnlLogger.cpp 
        	 tnlObject.cpp 
        	 tnlString.cpp 
        	 tnlTimerCPU.cpp  
        	 tnlTimerRT.cpp   
        	 mfilename.cpp 
        	 mpi-supp.cpp 
    )

SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlcoreinclude_HEADERS ${headers} )

ADD_LIBRARY( libtnlcore${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( libtnlcore${debugExt}-${tnlVersion} ${BZIP2_LIBRARIES} )

IF( BUILD_MPI )
    ADD_LIBRARY( libtnlcore-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( libtnlcore-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}
                                                                   ${BZIP2_LIBRARIES}  )
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core )
