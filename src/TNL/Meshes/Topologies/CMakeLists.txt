SET( headers Edge.h
             Hexahedron.h
             Quadrilateral.h
             Tetrahedron.h
             Triangle.h
             Vertex.h
             SubentityVertexMap.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/Topologies )
