SET( headers LaxFridrichs.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Operators/Advection )

   
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Operators/Advection )
