#if !defined(_TRIANGLE_H_)
#define _TRIANGLE_H_

#include "edge.h"


namespace topology
{


class Triangle
{
public:
	enum { dimension = 2 };
};


template<>
class BorderEntities<Triangle, 0>
{
public:
	typedef Vertex Tag;

	enum { count = 3 };
};

template<>
class BorderEntities<Triangle, 1>
{
public:
	typedef Edge Tag;

	enum { count = 3 };
};


template<> struct BorderEntityVertex<Triangle, Edge, 0, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Triangle, Edge, 0, 1> { enum { index = 2 }; };

template<> struct BorderEntityVertex<Triangle, Edge, 1, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Triangle, Edge, 1, 1> { enum { index = 0 }; };

template<> struct BorderEntityVertex<Triangle, Edge, 2, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Triangle, Edge, 2, 1> { enum { index = 1 }; };


} // namespace topology


#endif
