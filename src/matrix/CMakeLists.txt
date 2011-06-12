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

ADD_LIBRARY( libtnlmatrix${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers})

if( BUILD_MPI )
  target_link_libraries(libtnlmatrix${mpiExt}${debugExt}-0.1 mpi)
endif()

