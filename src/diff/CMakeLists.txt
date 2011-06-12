set( headers tnlMPIMesh.h
             tnlMPIMesh2D.h 
             tnlMPIMesh3D.h  
             curve-ident.h 
	)
set( sources )

set( libtnldiffincludedir ${TNL_INCLUDE_DIR}/diff )
set( libtnldiffinclude_HEADERS ${headers} )

add_library( libtnldiff${mpiExt}${debugExt}-0.1 SHARED ${sources} ${headers} ../dummy.cpp )	#${OPTIMISECXXFLAGS}

if( BUILD_MPI )
  target_link_libraries( libtnldiff-0.1 mpi )
endif()