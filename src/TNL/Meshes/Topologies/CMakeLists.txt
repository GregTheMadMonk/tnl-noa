SET( headers MeshEdgeTopology.h
             MeshEntityTopology.h
             MeshHexahedronTopology.h
             MeshQuadrilateralTopology.h
             MeshTetrahedronTopology.h
             MeshTriangleTopology.h
             MeshVertexTopology.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Meshes/Topologies )
