set( headers tnlDistributedGrid.h
             tnlGrid.h  
   )
set( sources )

set( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/mesh )
set( libtnlmeshinclude_HEADERS ${headers} )

add_library(libtnlmesh${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} ../dummy.cpp)	#${OPTIMISECXXFLAGS}

if (BUILD_MPI)
  target_link_libraries(libtnldiff-0.1 mpi)
endif()