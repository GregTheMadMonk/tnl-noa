SET( headers BufferEntitiesHelper.h
             DistributedMesh.h
             DistributedMeshSynchronizer.h
             DistributedGrid_1D.h
             DistributedGrid_2D.h
             DistributedGrid_3D.h
             DistributedGridSynchronizer.h
             DistributedGridIO.h
             )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/DistributedMeshes )
