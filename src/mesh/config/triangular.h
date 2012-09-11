#if !defined(_CONFIG_TRIANGULAR_H_)
#define _CONFIG_TRIANGULAR_H_

#include <mesh/topology/triangle.h>
#include <mesh/config/config_base.h>


namespace config
{


class TriangularMeshConfig : public MeshConfigBase
{
public:
	typedef topology::Triangle Cell;

	enum { dimension = Cell::dimension };
	enum { dimWorld = dimension };
};


} // namespace config


using config::TriangularMeshConfig;


#endif
