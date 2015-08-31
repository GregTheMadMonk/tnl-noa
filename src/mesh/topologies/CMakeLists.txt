SET( headers tnlMeshEdgeTopology.h
             tnlMeshEntityTopology.h
             tnlMeshHexahedronTopology.h
             tnlMeshQuadrilateralTopology.h
             tnlMeshTetrahedronTopology.h
             tnlMeshTriangleTopology.h
             tnlMeshVertexTopology.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/topologies )