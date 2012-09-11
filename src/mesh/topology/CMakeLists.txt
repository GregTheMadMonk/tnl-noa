SET( headers edge.h
             hexahedron.h
             quadrilateral.h
             tetrahedron.h
             topology.h
             triangle.h
             vertex.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/topology )