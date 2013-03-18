SET( headers tnlGrid1D_impl.h
             tnlGrid2D_impl.h
             tnlGrid3D_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/legacy/mesh/implementation )
set( tnl_legacy_mesh_implementation_SOURCES
     ${CURRENT_DIR}/tnlGridOld_impl.cpp 
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/legacy/mesh/implementation )