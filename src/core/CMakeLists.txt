ADD_SUBDIRECTORY( arrays )
ADD_SUBDIRECTORY( containers )
ADD_SUBDIRECTORY( cuda )
ADD_SUBDIRECTORY( vectors )

set (headers tnlAssert.h               
             tnlCurve.h
      	    tnlCuda.h
  		       tnlDataElement.h
  		       tnlDevice.h
  		       tnlDynamicTypeTag.h
  		       tnlFeature.h
  		       tnlFile.h 
  		       tnlFlopsCounter.h
   		    tnlHost.h 
   		    tnlIndexedSet.h
   		    tnlList.h
   		    tnlLogger.h 
   		    tnlObject.h 
   		    tnlStack.h
   		    tnlStaticFor.h
   		    tnlStatistics.h 
   		    tnlString.h 
   		    tnlReal.h
   		    tnlTimerCPU.h  
   		    tnlTimerRT.h    
   		    mfilename.h 
   		    mfuncs.h 
   		    mpi-supp.h 
   		    param-types.h
    )

SET( libtnlcoreincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlcoreinclude_HEADERS ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core )



