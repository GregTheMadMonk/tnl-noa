SET( headers BufferEntitiesHelper.h
             CopyEntitiesHelper.h
             DistributedMesh.h
             DistributedMeshSynchronizer.h
             DistributedGrid_1D.h
             DistributedGrid_1D.hpp
             DistributedGrid_2D.h
             DistributedGrid_2D.hpp
             DistributedGrid_3D.h
             DistributedGrid_3D.hpp
             DistributedGrid_Base.h
             DistributedGrid_Base.hpp
             DistributedGridSynchronizer.h
             DistributedGridSynchronizer_1D.h
             DistributedGridSynchronizer_2D.h
             DistributedGridSynchronizer_3D.h
             DistributedGridIO.h
             DistributedGridIO_MeshFunction.h
             DistributedGridIO_VectorField.h
             Directions.h
             SubdomainOverlapsGetter.h
             )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/DistributedMeshes )
