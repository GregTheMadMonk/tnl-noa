set( headers FlopsCounter.h
             Quad.h
             Quadcpp.h
             Real.h
              )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Experimental/Arithemtics )

set( tnl_experimental_arithmetics_SOURCES 
                 ${CURRENT_DIR}/FlopsCounter.cpp )

set( tnl_experimental_arithmetics_CUDA__SOURCES )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Experimental/Arithmetics )