ADD_SUBDIRECTORY( gradient )
ADD_SUBDIRECTORY( diffusion )
ADD_SUBDIRECTORY( euler )

SET( headers tnlFiniteDifferences.h
             tnlDirichletBoundaryConditions.h
             tnlNeumannBoundaryConditions.h
             tnlExactOperatorEvaluator.h )

#SET( libtnlmeshincludedir ${TNL_INCLUDE_DIR}/operators )
#SET( libtnlmeshinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators )