#if !defined(_CONFIG_QUADRILATERAL_H_)
#define _CONFIG_QUADRILATERAL_H_

#include <mesh/topology/quadrilateral.h>
#include <mesh/config/config_base.h>


namespace config
{


class QuadrilateralMeshConfig : public MeshConfigBase
{
public:
	typedef topology::Quadrilateral Cell;

	enum { dimension = Cell::dimension };
	enum { dimWorld = dimension };
};


} // namespace config


using config::QuadrilateralMeshConfig;


#endif
