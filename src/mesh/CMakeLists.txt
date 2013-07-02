#INCLUDE_DIRECTORIES( common config detail global io topology )

ADD_SUBDIRECTORY( common )
ADD_SUBDIRECTORY( config )
ADD_SUBDIRECTORY( detail )
ADD_SUBDIRECTORY( global )
ADD_SUBDIRECTORY( io )
ADD_SUBDIRECTORY( topology )

SET( headers tnlGrid.h
             tnlIdenticalGridGeometry.h
             tnlLinearGridGeometry.h
             tnlDummyMesh.h
             tnlGnuplotWriter.h
             config.h 
             information.h 
             mesh.h 
             traversal.h  )

#SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/mesh )
#SET( libtnlmeshinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh )