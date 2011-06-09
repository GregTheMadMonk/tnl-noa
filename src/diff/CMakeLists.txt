set (headers 
   mdiff-debug.h 
   drawGrid1D.h 
   drawGrid2D.h 
   drawGrid3D.h 
   tnlExplicitSolver.h 
   tnlGrid1D.h 
   tnlGrid2D.h 
   tnlGrid3D.h 
   tnlGridCUDA1D.h 
   tnlGridCUDA2D.h 
   tnlGridCUDA3D.h 
   tnlGridCUDA2DTester.h 
   tnlGridSystem1D.h 
   tnlEulerSolver.h 
   tnlMersonSolver.h 
   tnlMersonSolverCUDA.h 
   tnlMersonSolverCUDA.cu.h 
   tnlMPIMesh2D.h 
   tnlMPIMesh3D.h 
   tnlNonlinearRungeKuttaSolver.h 
   curve-ident.h 
   norms.h
	)
set (sources )

set (libtnldiffincludedir ${TNL_INCLUDE_DIR}/diff)
set (libtnldiffinclude_HEADERS ${headers})

if (BUILD_CUDA)
  CUDA_ADD_LIBRARY(libtnldiff${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} tnlMersonSolverCUDA.cu ../dummy.cpp)
  CUDA_ADD_LIBRARY(libdiff-tests${debugExt} SHARED tnlMersonSolverCUDATester.h tnlMersonSolverCUDATester.cu ../dummy.cpp)
else()
  ADD_LIBRARY(libtnldiff${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} ../dummy.cpp)	#${OPTIMISECXXFLAGS}
  ADD_LIBRARY(libdiff-tests${debugExt} SHARED ../dummy.cpp)
endif()
if (BUILD_MPI)
  target_link_libraries(libtnldiff-0.1 mpi)
endif()