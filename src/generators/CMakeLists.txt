ADD_SUBDIRECTORY( functions )

set (headers 
    )

SET( libtnlgeneratorsincludedir ${TNL_INCLUDE_DIR}/core )
SET( libtnlgeneratorsinclude_HEADERS ${headers} )
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/generators )

