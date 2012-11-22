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

	enum { dimWorld = Cell::dimension };
};


} // namespace config


using config::TriangularMeshConfig;


#endif
