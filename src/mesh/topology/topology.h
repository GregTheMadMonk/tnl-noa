#if !defined(_TOPOLOGY_H_)
#define _TOPOLOGY_H_

#include <mesh/common/common.h>


namespace topology
{


class Vertex;
class Edge;
class Triangle;
class Quadrilateral;
class Tetrahedron;
class Hexahedron;


template<typename MeshEntity, DimensionType entityBorderDimension> class BorderEntities;


template<typename MeshEntity, typename BorderEntity, int BorderEntityIndex, int BorderEntityVertexIndex> class BorderEntityVertex;


} // namespace topology


#endif
