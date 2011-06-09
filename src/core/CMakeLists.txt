set (headers tnlAssert.h  
    	     tnlCurve.h 
    	     tnlCudaSupport.h
		     tnlDataElement.h 
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
		     tnlRealTester.h 
		     tnlTimerCPU.h  
		     tnlTimerRT.h  
		     tnlVector.h 
		     tnl-cuda-kernels.h  
		     tnlCUDAKernelsTester.h
			 tnlCudaSupport.h
		     compress-file.h 
		     mfilename.h 
		     mfuncs.h 
		     mpi-supp.h 
		     param-types.h
    )
set( sources tnlCudaSupport.cpp 
             tnlFlopsCounter.cpp 
             tnlLogger.cpp 
        	 tnlObject.cpp 
        	 tnlString.cpp 
        	 tnlTimerCPU.cpp  
        	 tnlTimerRT.cpp   
        	 mfilename.cpp 
        	 mpi-supp.cpp 
    )

set (libmcoreincludedir ${TNL_INCLUDE_DIR}/core )
set (libmcoreinclude_HEADERS ${headers} )

if (BUILD_CUDA)
  CUDA_ADD_LIBRARY(libtnlcore${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} debug.h tnlLongVectorCUDA.cu tnl-cuda-kernels.cu )
else()
  ADD_LIBRARY(libtnlcore${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} debug.h)
endif()
if (BUILD_MPI)
  target_link_libraries(libtnlcore${mpiExt}${debugExt}-0.1 mpi)
endif()

