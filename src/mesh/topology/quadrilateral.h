#if !defined(_QUADRILATERAL_H_)
#define _QUADRILATERAL_H_

#include "edge.h"


namespace topology
{


class Quadrilateral
{
public:
	enum { dimension = 2 };
};


template<>
class BorderEntities<Quadrilateral, 0>
{
public:
	typedef Vertex Tag;

	enum { count = 4 };
};

template<>
class BorderEntities<Quadrilateral, 1>
{
public:
	typedef Edge Tag;

	enum { count = 4 };
};


template<> struct BorderEntityVertex<Quadrilateral, Edge, 0, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Quadrilateral, Edge, 0, 1> { enum { index = 1 }; };

template<> struct BorderEntityVertex<Quadrilateral, Edge, 1, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Quadrilateral, Edge, 1, 1> { enum { index = 2 }; };

template<> struct BorderEntityVertex<Quadrilateral, Edge, 2, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Quadrilateral, Edge, 2, 1> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Quadrilateral, Edge, 3, 0> { enum { index = 3 }; };
template<> struct BorderEntityVertex<Quadrilateral, Edge, 3, 1> { enum { index = 0 }; };


} // namespace topology


#endif
