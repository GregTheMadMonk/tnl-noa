SET( headers tnlAdaptiveRgCSRMatrix.h
             tnlCSRMatrix.h    
             tnlCusparseCSRMatrix.h     
	    	 tnlEllpackMatrix.h 
	         tnlEllpackMatrixCUDA.h 	        
	         tnlFastCSRMatrix.h 
	         tnlFastRgCSRMatrix.h
	         tnlFullMatrix.h
	         tnlMatrix.h
	         tnlMultiDiagonalMatrix.h
	         tnlPETSCMatrix.h
	         tnlRgCSRMatrix.h
	)
SET( sources tnlMatrix.cpp )

SET( libtnlmatrixincludedir  ${TNL_INCLUDE_DIR}/matrix )
SET( libtnlmatrixinclude_HEADERS ${headers} )

ADD_LIBRARY( tnlmatrix${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
TARGET_LINK_LIBRARIES( tnlmatrix${debugExt}-${tnlVersion} )
INSTALL( TARGETS tnlmatrix${debugExt}-${tnlVersion} DESTINATION lib )

IF( BUILD_MPI )
    ADD_LIBRARY( tnlmatrix-mpi${debugExt}-${tnlVersion} SHARED ${sources} ${headers} )
    TARGET_LINK_LIBRARIES( tnlmatrix-mpi${debugExt}-${tnlVersion} ${MPI_LIBRARIES}  )
    SET_TARGET_PROPERTIES( tnlmatrix-mpi${debugExt}-${tnlVersion} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
    INSTALL( TARGETS tnlmatrix-mpi${debugExt}-${tnlVersion} DESTINATION lib )
ENDIF()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/matrix )