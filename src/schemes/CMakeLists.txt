ADD_SUBDIRECTORY( gradient )
ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )

SET( headers tnlFiniteDifferences.h  )

#SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/schemes )
#SET( libtnlmeshinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/schemes )