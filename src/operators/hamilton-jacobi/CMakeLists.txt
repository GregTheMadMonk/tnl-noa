ADD_SUBDIRECTORY( godunov-eikonal )

set( headers godunovEikonal.h
	     upwindEikonal.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/operators/hamilton-jacobi )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/operators/hamilton-jacobi )
