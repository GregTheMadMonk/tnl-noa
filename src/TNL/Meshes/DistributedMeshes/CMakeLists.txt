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
             DistributedGridIO.h
             Directions.h
             )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/DistributedMeshes )
