INCLUDE_DIRECTORIES( common config detail global io topology )

ADD_SUBDIRECTORY( common )
ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( detail )
ADD_SUBDIRECTORY( global )
ADD_SUBDIRECTORY( io )
ADD_SUBDIRECTORY( topology )

SET( headers tnlDistributedGrid.h
             tnlGrid.h  
             config.h 
             information.h 
             mesh.h 
             traversal.h  )

#SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/mesh )
#SET( libtnlmeshinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )