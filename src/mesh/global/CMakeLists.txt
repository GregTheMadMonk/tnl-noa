SET( headers array.h
             static_array.h
             static_assert.h
             static_if.h
             static_loop.h
             static_switch.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/global )
