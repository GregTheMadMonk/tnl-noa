SET( headers tnlGrid1D_impl.h
             tnlGrid2D_impl.h
             tnlGrid3D_impl.h  )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/mesh )