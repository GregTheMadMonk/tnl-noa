SET( headers LaxFridrichs.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators/Advection )

   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Operators/Advection )
