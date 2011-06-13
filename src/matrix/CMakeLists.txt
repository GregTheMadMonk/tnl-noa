SET( headers tnlAdaptiveRgCSRMatrix.h
             tnlCSRMatrix.h         
	    	 tnlEllpackMatrix.h 
	         tnlEllpackMatrixCUDA.h 	        
	         tnlFastCSRMatrix.h 
	         tnlFastRgCSRMatrix.h
	         tnlFullMatrix.h
	         tnlMatrix.h
	         tnlPETSCMatrix.h
	         tnlRgCSRMatrix.h
	)
SET( sources tnlMatrix.cpp )

SET( libtnlmatrixincludedir  ${TNL_INCLUDE_DIR}/matrix )
SET( libtnlmatrixinclude_HEADERS ${headers} )

ADD_LIBRARY( libtnlmatrix${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( libtnlmatrix${debugExt}-${tnlVersion} )

IF( BUILD_MPI )
    ADD_LIBRARY( libtnlmatrix-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( libtnlmatrix-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    ADD_DEFINITIONS( ${MPI_COMPILE_FLAGS} )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/matrix )