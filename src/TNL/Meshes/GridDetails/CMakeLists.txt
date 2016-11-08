SET( headers BoundaryGridEntityChecker.h
             BoundaryGridEntityChecker.h
             GnuplotWriter.h
             Grid1D.h
             Grid1D_impl.h
             Grid2D.h
             Grid2D_impl.h
             Grid3D.h
             Grid3D_impl.h
             GridEntityCenterGetter.h
             GridEntityGetter.h
             GridEntityGetter_impl.h
             GridEntity_impl.h
             GridEntityMeasureGetter.h
             GridEntityTopology.h
             GridTraverser.h
             GridTraverser_impl.h
             NeighbourGridEntitiesStorage.h
             NeighbourGridEntityGetter1D_impl.h
             NeighbourGridEntityGetter2D_impl.h
             NeighbourGridEntityGetter3D_impl.h
             NeighbourGridEntityGetter.h
             Traverser_Grid1D.h
             Traverser_Grid1D_impl.h
             Traverser_Grid2D.h
             Traverser_Grid2D_impl.h
             Traverser_Grid3D.h
             Traverser_Grid3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Meshes/GridDetails )    
SET( tnl_mesh_griddetails_SOURCES
     ${CURRENT_DIR}/Grid_impl.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/GridDetails )
