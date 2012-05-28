SET( headers tnlDistributedGrid.h
             tnlGrid.h )

#SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/mesh )
#SET( libtnlmeshinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )