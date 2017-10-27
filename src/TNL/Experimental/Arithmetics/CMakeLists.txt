ADD_SUBDIRECTORY( UnitTests )

set( headers FlopsCounter.h
             QuadDouble.h # Edited by D. Simon
             Real.h
	     MultiPrecision.h #Added by D. Simon
	     Multiple.h #Added by D. Simon
              )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Experimental/Arithemtics )

set( tnl_experimental_arithmetics_SOURCES 
                 ${CURRENT_DIR}/FlopsCounter.cpp
		 ${CURRENT_DIR}/MultiPrecision.cpp #Added by D.Simon
		 ${CURRENT_DIR}/QuadDouble.cpp #Added by D.Simon

set( tnl_experimental_arithmetics_CUDA__SOURCES )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Experimental/Arithmetics )
