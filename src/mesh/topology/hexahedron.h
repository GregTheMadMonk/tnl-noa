#if !defined(_HEXAHEDRON_H_)
#define _HEXAHEDRON_H_

#include "quadrilateral.h"


namespace topology
{


class Hexahedron
{
public:
	enum { dimension = 3 };
};


template<>
class BorderEntities<Hexahedron, 0>
{
public:
	typedef Vertex Tag;

	enum { count = 8 };
};

template<>
class BorderEntities<Hexahedron, 1>
{
public:
	typedef Edge Tag;

	enum { count = 12 };
};

template<>
class BorderEntities<Hexahedron, 2>
{
public:
	typedef Quadrilateral Tag;

	enum { count = 6 };
};


template<> struct BorderEntityVertex<Hexahedron, Edge,  0, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  0, 1> { enum { index = 1 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  1, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  1, 1> { enum { index = 2 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  2, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  2, 1> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  3, 0> { enum { index = 3 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  3, 1> { enum { index = 0 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  4, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  4, 1> { enum { index = 4 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  5, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  5, 1> { enum { index = 5 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  6, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  6, 1> { enum { index = 6 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  7, 0> { enum { index = 3 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  7, 1> { enum { index = 7 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  8, 0> { enum { index = 4 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  8, 1> { enum { index = 5 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge,  9, 0> { enum { index = 5 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge,  9, 1> { enum { index = 6 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge, 10, 0> { enum { index = 6 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge, 10, 1> { enum { index = 7 }; };

template<> struct BorderEntityVertex<Hexahedron, Edge, 11, 0> { enum { index = 7 }; };
template<> struct BorderEntityVertex<Hexahedron, Edge, 11, 1> { enum { index = 4 }; };


template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 0, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 0, 1> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 0, 2> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 0, 3> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 1, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 1, 1> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 1, 2> { enum { index = 5 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 1, 3> { enum { index = 4 }; };

template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 2, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 2, 1> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 2, 2> { enum { index = 6 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 2, 3> { enum { index = 5 }; };

template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 3, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 3, 1> { enum { index = 3 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 3, 2> { enum { index = 7 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 3, 3> { enum { index = 6 }; };

template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 4, 0> { enum { index = 3 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 4, 1> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 4, 2> { enum { index = 4 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 4, 3> { enum { index = 7 }; };

template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 5, 0> { enum { index = 4 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 5, 1> { enum { index = 5 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 5, 2> { enum { index = 6 }; };
template<> struct BorderEntityVertex<Hexahedron, Quadrilateral, 5, 3> { enum { index = 7 }; };


} // namespace topology


#endif
