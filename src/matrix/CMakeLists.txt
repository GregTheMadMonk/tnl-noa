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
	
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/matrix )
SET( tnl_matrix_SOURCES 
     ${CURRENT_DIR}/tnlMatrix.cpp 
     PARENT_SCOPE )

#SET( libtnlmatrixincludedir  ${TNL_INCLUDE_DIR}/matrix )
#SET( libtnlmatrixinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/matrix )