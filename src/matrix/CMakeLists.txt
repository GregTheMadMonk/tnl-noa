set (headers 
     tnlMatrix.h 
	  tnlCSRMatrixOld.h 
	  tnlCoalescedCSRMatrix.h 
	  tnlCoalescedCSRMatrixCUDA.h 
	  tnlCoalescedFastCSRMatrix.h 
	  tnlCoalescedFastCSRMatrixCUDA.h 
	  tnlEllpackMatrix.h 
	  tnlEllpackMatrixCUDA.h 
	  tnlCSRMatrix.h 
	  tnlCSRMatrixTester.h 
	  tnlFastCSRMatrix.h 
	  tnlFullMatrix.h 
	  tnlBaseMatrix.h 
	  tnlCoalescedCSRMatrixTester.h 
	  tnlEllpackMatrixTester.h 
	  tnlMatrixSolver.h 
	  tnlPETSCMatrix.h 
	  tnlPETSCSolver.h 
	  tnlSORSolver.h 
	  tnlCGSolver.h 
	  tnlBICGSolver.h 
	  tnlBICGStabSolver.h 
	  tnlGMRESSolver.h  
	  tnlPreconditioner.h 
	  tnlILUPreconditioner.h 
	  tnlPETSCPreconditioner.h
	)
set (sources tnlBaseMatrix.cpp)

set (libtnlmatrixincludedir  ${TNL_INCLUDE_DIR}/matrix)
set (libtnlmatrixinclude_HEADERS ${headers})

if (BUILD_CUDA)
  set (cudaAdd tnlCoalescedCSRMatrixCUDA.cu
	    tnlCoalescedFastCSRMatrixCUDA.cu
	    tnlEllpackMatrixCUDA.cu
	    tnlCoalescedCSRMatrixCUDA.cu.h
	    tnlCoalescedFastCSRMatrixCUDA.cu.h
	    tnlEllpackMatrixCUDA.cu.h 
  )
  CUDA_ADD_LIBRARY(libtnlmatrix${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} ${cudaAdd}) # ${OPTIMISECXXFLAGS}
else()
  ADD_LIBRARY(libtnlmatrix${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers}) # ${OPTIMISECXXFLAGS}
endif()
if (BUILD_MPI)
  target_link_libraries(libtnlmatrix${mpiExt}${debugExt}-0.1 mpi)
endif()

