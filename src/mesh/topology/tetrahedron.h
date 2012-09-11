#if !defined(_TETRAHEDRON_H_)
#define _TETRAHEDRON_H_

#include "triangle.h"


namespace topology
{


class Tetrahedron
{
public:
	enum { dimension = 3 };
};


template<>
class BorderEntities<Tetrahedron, 0>
{
public:
	typedef Vertex Tag;

	enum { count = 4 };
};

template<>
class BorderEntities<Tetrahedron, 1>
{
public:
	typedef Edge Tag;

	enum { count = 6 };
};

template<>
class BorderEntities<Tetrahedron, 2>
{
public:
	typedef Triangle Tag;

	enum { count = 4 };
};


template<> struct BorderEntityVertex<Tetrahedron, Edge, 0, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Tetrahedron, Edge, 0, 1> { enum { index = 2 }; };

template<> struct BorderEntityVertex<Tetrahedron, Edge, 1, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Tetrahedron, Edge, 1, 1> { enum { index = 0 }; };

template<> struct BorderEntityVertex<Tetrahedron, Edge, 2, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Tetrahedron, Edge, 2, 1> { enum { index = 1 }; };

template<> struct BorderEntityVertex<Tetrahedron, Edge, 3, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Tetrahedron, Edge, 3, 1> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Tetrahedron, Edge, 4, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Tetrahedron, Edge, 4, 1> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Tetrahedron, Edge, 5, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Tetrahedron, Edge, 5, 1> { enum { index = 3 }; };


template<> struct BorderEntityVertex<Tetrahedron, Triangle, 0, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 0, 1> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 0, 2> { enum { index = 2 }; };

template<> struct BorderEntityVertex<Tetrahedron, Triangle, 1, 0> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 1, 1> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 1, 2> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Tetrahedron, Triangle, 2, 0> { enum { index = 1 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 2, 1> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 2, 2> { enum { index = 3 }; };

template<> struct BorderEntityVertex<Tetrahedron, Triangle, 3, 0> { enum { index = 2 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 3, 1> { enum { index = 0 }; };
template<> struct BorderEntityVertex<Tetrahedron, Triangle, 3, 2> { enum { index = 3 }; };


} // namespace topology


#endif
