SET( headers tnlMeshEdgeTag.h
             tnlMeshEntityTopology.h
             tnlMeshHexahedronTag.h
             tnlMeshQuadrilateralTag.h
             tnlMeshTetrahedronTag.h
             tnlMeshTriangleTag.h
             tnlMeshVertexTag.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/topologies )