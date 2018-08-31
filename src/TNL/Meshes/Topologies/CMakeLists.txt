SET( headers Edge.h
             Hexahedron.h
             Quadrilateral.h
             Simplex.h
             SubentityVertexMap.h
             Tetrahedron.h
             Triangle.h
             Vertex.h
)

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Meshes/Topologies )
