SET( headers tnlGrid1D_impl.h
             tnlGrid2D_impl.h
             tnlGrid3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/mesh )    
SET( tnl_implementation_mesh_SOURCES
     ${CURRENT_DIR}/tnlGrid_impl.cpp
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/mesh )