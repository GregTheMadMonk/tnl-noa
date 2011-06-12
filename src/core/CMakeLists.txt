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

set( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
set( libtnlcoreinclude_HEADERS ${headers} )

ADD_LIBRARY(libtnlcore${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} )

if (BUILD_MPI)
  target_link_libraries(libtnlcore${mpiExt}${debugExt}-0.1 mpi
                                                            ${BZIP2_LIBRARIES}  )
else()
    target_link_libraries(libtnlcore${mpiExt}${debugExt}-0.1 ${BZIP2_LIBRARIES} )
endif()

