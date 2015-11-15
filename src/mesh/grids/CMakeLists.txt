SET( headers tnlGrid1D.h
             tnlGrid1D_impl.h
             tnlGrid2D.h
             tnlGrid2D_impl.h
             tnlGrid3D.h
             tnlGrid3D_impl.h
             tnlGridEntityTopology.h
             tnlGridEntityCenterGetter.h
             tnlGridEntityCenterGetter_impl.h
             tnlTraverser_Grid1D.h
             tnlTraverser_Grid1D_impl.h 
             tnlTraverser_Grid2D.h
             tnlTraverser_Grid2D_impl.h 
             tnlTraverser_Grid3D.h
             tnlTraverser_Grid3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/mesh/grids )    
SET( tnl_mesh_grids_SOURCES
     ${CURRENT_DIR}/tnlGrid_impl.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/grids )