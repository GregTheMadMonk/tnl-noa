set( ENABLE_CODECOVERAGE )

ADD_SUBDIRECTORY( data )
ADD_SUBDIRECTORY( benchmarks )
ADD_SUBDIRECTORY( approximation-tests )
ADD_SUBDIRECTORY( unit-tests )
ADD_SUBDIRECTORY( mesh )

SET( headers matrix-formats-test.h )

#ADD_EXECUTABLE( tnl-test-matrix-formats${mpiExt}${debugExt} ${headers} matrix-formats-test.cpp )

unset( ENABLE_CODECOVERAGE )