SET( headers Sign.h
             Heaviside.h
             Identity.h
             SmoothHeaviside.h
             Shift.h
             Rotation.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators/Analytic )

   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Operators/Analytic )
