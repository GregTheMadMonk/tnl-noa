SET( headers MeshEdgeTopology.h
             MeshEntityTopology.h
             MeshHexahedronTopology.h
             MeshQuadrilateralTopology.h
             MeshTetrahedronTopology.h
             MeshTriangleTopology.h
             MeshVertexTopology.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/Topologies )
