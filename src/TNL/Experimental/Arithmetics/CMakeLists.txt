ADD_SUBDIRECTORY( UnitTests )

set( headers FlopsCounter.h
             Quad.h # Modified by D. Simon
             Quad_impl.h # Modified by D. Simon  
             Real.h
	     MultiPrecision.h #Added by D. Simon
	     Double.h #Added by D. Simon
	     Double_impl.h #Added by D. Simon
              )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Experimental/Arithmetics )

set( tnl_experimental_arithmetics_SOURCES 
                 ${CURRENT_DIR}/FlopsCounter.cpp
		 ${CURRENT_DIR}/MultiPrecision.cpp #Added by D.Simon
		 )

set( tnl_experimental_arithmetics_CUDA__SOURCES )

# INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Experimental/Arithmetics ) old
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Experimental/Arithmetics )
