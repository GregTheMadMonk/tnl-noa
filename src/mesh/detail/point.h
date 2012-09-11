#if !defined(_POINT_H_)
#define _POINT_H_

#include <mesh/global/static_array.h>
#include <mesh/common/common.h>


namespace implementation
{


template<typename CoordType, DimensionType dim>
class PointBase
{
public:
	enum { dimension = dim };

	CoordType &operator[](DimensionType index)             { assert(0 <= index && index < dimension); return m_coords[index]; }
	const CoordType &operator[](DimensionType index) const { assert(0 <= index && index < dimension); return m_coords[index]; }

protected:
	StaticArray<CoordType, DimensionType, dim> m_coords;
};


template<typename CoordType, DimensionType dim>
class Point : public PointBase<CoordType, dim>
{
};


template<typename CoordType>
class Point<CoordType, 1> : public PointBase<CoordType, 1>
{
public:
	Point() {}

	Point(CoordType x1)
	{
		setCoords(x1);
	}

	void setCoords(CoordType x1)
	{
		this->m_coords[0] = x1;
	}
};


template<typename CoordType>
class Point<CoordType, 2> : public PointBase<CoordType, 2>
{
public:
	Point() {}

	Point(CoordType x1, CoordType x2)
	{
		setCoords(x1, x2);
	}

	void setCoords(CoordType x1, CoordType x2)
	{
		this->m_coords[0] = x1;
		this->m_coords[1] = x2;
	}
};


template<typename CoordType>
class Point<CoordType, 3> : public PointBase<CoordType, 3>
{
public:
	Point() {}

	Point(CoordType x1, CoordType x2, CoordType x3)
	{
		setCoords(x1, x2, x3);
	}

	void setCoords(CoordType x1, CoordType x2, CoordType x3)
	{
		this->m_coords[0] = x1;
		this->m_coords[1] = x2;
		this->m_coords[2] = x3;
	}
};


} // namespace implementation


#endif
