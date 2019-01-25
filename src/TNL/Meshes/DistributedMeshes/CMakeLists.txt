SET( headers BufferEntitiesHelper.h
             CopyEntitiesHelper.h
             DistributedMesh.h
             DistributedMeshSynchronizer.h
             DistributedGrid.h
             DistributedGrid.hpp
             DistributedGridSynchronizer.h
             DistributedGridIO.h
             DistributedGridIO_MeshFunction.h
             DistributedGridIO_VectorField.h
             Directions.h
             SubdomainOverlapsGetter.h
             SubdomainOverlapsGetter.hpp
             )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/DistributedMeshes )
