#if !defined(_EDGE_H_)
#define _EDGE_H_

#include "topology.h"
#include "vertex.h"


namespace topology
{


class Edge
{
public:
	enum { dimension = 1 };
};


template<>
class BorderEntities<Edge, 0>
{
public:
	typedef Vertex Tag;

	enum { count = 2 };
};


} // namespace topology


#endif
