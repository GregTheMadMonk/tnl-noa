ADD_SUBDIRECTORY( gradient )
ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )
ADD_SUBDIRECTORY( godunov )
ADD_SUBDIRECTORY( upwind )
ADD_SUBDIRECTORY( godunov-eikonal )
ADD_SUBDIRECTORY( upwind-eikonal )

SET( headers tnlFiniteDifferences.h
             tnlDirichletBoundaryConditions.h
             tnlNeumannBoundaryConditions.h
	     tnlNeumannReflectionBoundaryConditions.h )

#SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/operators )
#SET( libtnlmeshinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators )
