#if !defined(_TEST_CONFIGS_H_)
#define _TEST_CONFIGS_H_

#include <mesh/config.h>


namespace config
{


class Lin3DConfig : public LinearMeshConfig
{
public:
	typedef topology::Edge Cell;

	enum { dimension = Cell::dimension };
	enum { dimWorld = 3 };
};

template<> struct CoborderStorage<Lin3DConfig, topology::Vertex, 1> { enum { enabled = true }; };



class Tri2DAllConfig : public TriangularMeshConfig
{
};

template<> struct CoborderStorage<Tri2DAllConfig, topology::Vertex, 1> { enum { enabled = true }; };
template<> struct CoborderStorage<Tri2DAllConfig, topology::Vertex, 2> { enum { enabled = true }; };
template<> struct CoborderStorage<Tri2DAllConfig, topology::Edge,   2> { enum { enabled = true }; };



class Quadri2DCoborderWithoutBorderConfig : public QuadrilateralMeshConfig
{
};

template<> struct BorderStorage<Quadri2DCoborderWithoutBorderConfig, topology::Quadrilateral, 1> { enum { enabled = false }; };

template<> struct CoborderStorage<Quadri2DCoborderWithoutBorderConfig, topology::Edge, 2> { enum { enabled = true }; };



class Hexa3DNothingConfig : public HexahedralMeshConfig
{
};

template<> struct EntityStorage<Hexa3DNothingConfig, 1> { enum { enabled = false }; };
template<> struct EntityStorage<Hexa3DNothingConfig, 2> { enum { enabled = false }; };



class Hexa3DAllConfig : public HexahedralMeshConfig
{
};

template<> struct CoborderStorage<Hexa3DAllConfig, topology::Vertex,        1> { enum { enabled = true }; };
template<> struct CoborderStorage<Hexa3DAllConfig, topology::Vertex,        2> { enum { enabled = true }; };
template<> struct CoborderStorage<Hexa3DAllConfig, topology::Vertex,        3> { enum { enabled = true }; };
template<> struct CoborderStorage<Hexa3DAllConfig, topology::Edge,          2> { enum { enabled = true }; };
template<> struct CoborderStorage<Hexa3DAllConfig, topology::Edge,          3> { enum { enabled = true }; };
template<> struct CoborderStorage<Hexa3DAllConfig, topology::Quadrilateral, 3> { enum { enabled = true }; };


} // namespace config


#endif // !defined(_TEST_CONFIGS_H_)
