SET( headers tnlDenseMatrix.h
             tnlTridiagonalMatrix.h

             tnlAdaptiveRgCSRMatrix.h
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
	
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/matrices )
set( common_SOURCES ${CURRENT_DIR}/tnlMatrix.cpp )

SET( tnl_matrices_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
   SET( tnl_matrices_CUDA__SOURCES 
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/matrices )